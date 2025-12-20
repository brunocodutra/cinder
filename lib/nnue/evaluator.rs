use crate::nnue::{Accumulator, Bucket, Feature, Nnue, Synapse, Value};
use crate::util::{Assume, Float, Int};
use crate::{chess::*, params::Params, search::Ply, simd::*};
use bytemuck::zeroed;
use derive_more::with_trait::Debug;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Index, Range};
use std::{array, hint::unreachable_unchecked, str::FromStr};

#[cfg(test)]
use proptest::{prelude::*, sample::*};

#[derive(Debug, Clone, Hash)]
#[derive_const(Eq, PartialEq)]
struct CachedAccumulator {
    accumulator: Accumulator,
    pieces: Aligned<[Option<Piece>; Square::MAX as usize + 1]>,
    occupied: Bitboard,
}

impl const Default for CachedAccumulator {
    fn default() -> Self {
        let mut cache = CachedAccumulator {
            accumulator: zeroed(),
            pieces: Aligned([None; Square::MAX as usize + 1]),
            occupied: Bitboard::empty(),
        };

        Nnue::transformer().refresh(&mut cache.accumulator);
        cache
    }
}

#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq)]
#[repr(u8)]
enum Pending {
    Update,
    Refresh,
}

/// A [`Position`] evaluation stack.
#[derive(Debug, Clone)]
#[debug("Evaluator({})", self.deref())]
pub struct Evaluator {
    ply: Ply,
    positions: [Position; Ply::MAX as usize + 1],
    accumulator: [[Accumulator; 2]; Ply::MAX as usize + 1],
    pending: [[Option<Pending>; Ply::MAX as usize + 1]; 2],
    // move[i] leads to pos[i + 1]
    moves: [Option<Move>; Ply::MAX as usize],
    cache: [[CachedAccumulator; Bucket::LEN]; 2],
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new(Position::default())
    }
}

impl const Eq for Evaluator {}

impl const PartialEq for Evaluator {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other)
    }
}

impl Hash for Evaluator {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl const Deref for Evaluator {
    type Target = Position;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.index(self.ply)
    }
}

impl const Index<Ply> for Evaluator {
    type Output = Position;

    #[inline(always)]
    fn index(&self, ply: Ply) -> &Self::Output {
        self.positions.get(ply.cast::<usize>()).assume()
    }
}

#[cfg(test)]
impl Arbitrary for Evaluator {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (any::<Ply>(), any::<Selector>(), any::<Position>())
            .prop_map(|(plies, selector, pos)| {
                let mut pos = Evaluator::new(pos);

                for _ in 0..plies.cast::<usize>() {
                    if pos.outcome().is_none() {
                        pos.push(selector.try_select(pos.moves().unpack()));
                    } else {
                        break;
                    }
                }

                pos
            })
            .no_shrink()
            .boxed()
    }
}

impl Evaluator {
    /// Constructs the evaluator from a [`Position`].
    pub fn new(pos: Position) -> Self {
        let mut evaluator = Evaluator {
            ply: Ply::new(0),
            positions: array::from_fn(|_| pos.clone()),
            accumulator: zeroed(),
            pending: [[None; Ply::MAX as usize + 1]; 2],
            moves: [None; Ply::MAX as usize],
            cache: Default::default(),
        };

        evaluator.reset();
        evaluator
    }

    /// The current [`Ply`].
    #[inline(always)]
    pub fn ply(&self) -> Ply {
        self.ply
    }

    /// Piece values at this phase of the game.
    #[inline(always)]
    pub fn piece_values(&self) -> [f32; Role::MAX as usize + 1] {
        let phase = self.phase().cast::<usize>();

        [
            *Params::pawn_values(phase),
            *Params::knight_values(phase),
            *Params::bishop_values(phase),
            *Params::rook_values(phase),
            *Params::queen_values(phase),
            0.,
        ]
    }

    /// Estimates the material gain of a move.
    #[inline(always)]
    pub fn gain(&self, m: Move) -> Value {
        let mut gain = 0.;

        if !m.is_quiet() {
            let piece_values = self.piece_values();

            if let Some(victim) = self.role_on(m.whither()) {
                gain += piece_values[victim.cast::<usize>()];
            } else if m.is_capture() {
                gain += piece_values[Role::Pawn.cast::<usize>()];
            }

            if let Some(promotion) = m.promotion() {
                gain += piece_values[promotion.cast::<usize>()];
                gain -= piece_values[Role::Pawn.cast::<usize>()];
            }
        }

        gain.to_int()
    }

    /// Whether this move wins the exchange by at least `margin`.
    #[inline(always)]
    pub fn winning(&self, m: Move, margin: Value) -> bool {
        margin == Value::lower() || self.see(m, margin - 1..margin) == margin
    }

    /// Computes the static exchange evaluation.
    #[inline(always)]
    pub fn see(&self, m: Move, bounds: Range<Value>) -> Value {
        let (mut alpha, mut beta) = (bounds.start, bounds.end);
        let mut score = self.gain(m);
        beta = beta.min(score);

        if alpha >= beta {
            return alpha;
        }

        let piece_values: [i16; _] = self.piece_values().map(Float::to_int);

        score -= match m.promotion() {
            None => piece_values[self.role_on(m.whence()).assume().cast::<usize>()],
            Some(promotion) => piece_values[promotion.cast::<usize>()],
        };

        alpha = alpha.max(score);

        if alpha >= beta {
            return beta;
        }

        let mut exchanges = self.exchanges(m);

        loop {
            let Some((_, captor, _)) = exchanges.next() else {
                break beta;
            };

            score = -(score + piece_values[captor.cast::<usize>()]);
            beta = beta.min(-score);

            if alpha >= beta {
                break alpha;
            }

            let Some((_, captor, _)) = exchanges.next() else {
                break alpha;
            };

            score = -(score + piece_values[captor.cast::<usize>()]);
            alpha = alpha.max(score);

            if alpha >= beta {
                break beta;
            }
        }
    }

    /// Pushes a [`Position`] into the evaluator stack.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn push(&mut self, m: Option<Move>) {
        (self.ply < Ply::MAX).assume();

        let turn = self.turn();
        let idx = self.ply.cast::<usize>();

        self.ply += 1;
        self.moves[idx] = m;
        self.pending[0][idx + 1] = Some(Pending::Update);
        self.pending[1][idx + 1] = Some(Pending::Update);
        self.positions[idx + 1] = self.positions[idx].clone();

        match self.moves[idx] {
            None => self.positions[idx + 1].pass(),
            Some(m) => {
                self.positions[idx + 1].play(m);
                let (wc, wt) = (m.whence(), m.whither());
                let role = self.positions[idx].role_on(m.whence()).assume();
                if role == Role::King && Feature::bucket(turn, wc) != Feature::bucket(turn, wt) {
                    self.pending[turn.cast::<usize>()][idx + 1] = Some(Pending::Refresh)
                }
            }
        }
    }

    /// Pops a [`Position`] from the evaluator stack.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pop(&mut self) {
        (self.ply > 0).assume();
        self.ply -= 1;
    }

    /// Resets the evaluator stack from the current [`Position`].
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn reset(&mut self) {
        for side in Color::iter() {
            self.refresh(side);
        }

        if self.ply > 0 {
            let idx = self.ply.cast::<usize>();
            unsafe { self.positions.swap_unchecked(0, idx) };
            unsafe { self.accumulator.swap_unchecked(0, idx) };
            self.pending = [[None; Ply::MAX as usize + 1]; 2];
            self.moves = [None; Ply::MAX as usize];
            self.ply = Ply::lower();
        }
    }

    /// Evaluates the [`Position`] at the current [`Ply`].
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn evaluate(&mut self) -> Value {
        for side in Color::iter() {
            let mut idx = self.ply.cast::<usize>();
            if self.pending[side.cast::<usize>()][idx].is_none() {
                continue;
            }

            while self.pending[side.cast::<usize>()][idx] == Some(Pending::Update) {
                idx = idx.checked_sub(1).assume();
            }

            match self.pending[side.cast::<usize>()][idx] {
                Some(Pending::Update) => unsafe { unreachable_unchecked() },
                Some(Pending::Refresh) => self.refresh(side),
                None => {
                    for i in idx + 1..=self.ply.cast::<usize>() {
                        self.update(side, i.convert().assume());
                    }
                }
            }
        }

        let phase = self.phase();
        let nn = Nnue::nn(phase);

        let idx = self.ply.cast::<usize>();
        debug_assert_eq!(self.pending[0][idx], None);
        debug_assert_eq!(self.pending[1][idx], None);

        let us = self.turn() as usize;
        let them = self.turn().flip() as usize;
        nn.forward((&self.accumulator[idx][us], &self.accumulator[idx][them]))
    }

    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn refresh(&mut self, side: Color) {
        let idx = self.ply.cast::<usize>();
        let pos = &self.positions[idx];
        let ksq = pos.king(side);
        let bucket = Feature::bucket(side, ksq).cast::<usize>();

        const N: usize = Square::MAX as usize + 1;
        let cache = &self.cache[side.cast::<usize>()][bucket];
        let current: &Simd<u8, N> = cache.pieces.cast();
        let target: &Simd<u8, N> = pos.pieces().cast();
        let diff = Bitboard(current.simd_ne(*target).to_bitmask());

        let mut to_sub = Squares::new(cache.occupied & diff);
        let mut to_add = Squares::new(pos.occupied() & diff);

        while to_sub.len() > 0 || to_add.len() > 0 {
            let sub = array::from_fn(|_| {
                to_sub.next().map(|sq| {
                    let cache = &self.cache[side.cast::<usize>()][bucket];
                    let piece = cache.pieces[sq.cast::<usize>()].assume();
                    Feature::new(side, ksq, piece, sq)
                })
            });

            let add = array::from_fn(|_| {
                to_add.next().map(|sq| {
                    let piece = pos.piece_on(sq).assume();
                    Feature::new(side, ksq, piece, sq)
                })
            });

            let cache = &mut self.cache[side.cast::<usize>()][bucket];
            Nnue::transformer().accumulate_in_place(&mut cache.accumulator, sub, add);
        }

        self.cache[side.cast::<usize>()][bucket].occupied = pos.occupied();
        self.cache[side.cast::<usize>()][bucket].pieces = *pos.pieces();

        self.pending[side.cast::<usize>()][idx] = None;
        self.accumulator[idx][side.cast::<usize>()] =
            self.cache[side.cast::<usize>()][bucket].accumulator.clone();
    }

    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, side: Color, ply: Ply) {
        (ply > 0).assume();

        let idx = ply.cast::<usize>();
        self.pending[side.cast::<usize>()][idx] = None;

        let mut sub = [None; 2];
        let mut add = [None; 2];

        if let Some(m) = self.moves[idx - 1] {
            let pos = &self.positions[idx];
            let prev = &self.positions[idx - 1];
            let (wc, wt) = (m.whence(), m.whither());
            let promotion = m.promotion();
            let role = prev.role_on(wc).assume();
            let turn = prev.turn();

            let ksq = pos.king(side);
            let old = Piece::new(role, turn);
            let new = Piece::new(promotion.unwrap_or(role), turn);
            sub[0] = Some(Feature::new(side, ksq, old, wc));
            add[0] = Some(Feature::new(side, ksq, new, wt));

            let capture = match prev.role_on(wt) {
                _ if !m.is_capture() => None,
                Some(r) => Some((r, wt)),
                None => Some((Role::Pawn, Square::new(wt.file(), wc.rank()))),
            };

            if let Some((r, sq)) = capture {
                let victim = Piece::new(r, !turn);
                sub[1] = Some(Feature::new(side, ksq, victim, sq));
            } else if role == Role::King && (wt - wc).abs() == 2 {
                let m = Castles::rook(wt).assume();
                let rook = Piece::new(Role::Rook, turn);
                sub[1] = Some(Feature::new(side, ksq, rook, m.whence()));
                add[1] = Some(Feature::new(side, ksq, rook, m.whither()));
            }
        }

        let (left, right) = self.accumulator.split_at_mut(idx);
        let src = &left[left.len() - 1][side.cast::<usize>()];
        let dst = &mut right[0][side.cast::<usize>()];
        Nnue::transformer().accumulate(src, dst, sub, add);
    }
}

impl FromStr for Evaluator {
    type Err = ParsePositionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s.parse()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::sample::select;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn evaluator_updates_accumulator_lazily(
        #[filter(#pos.outcome().is_none())] mut pos: Evaluator,
    ) {
        assert_eq!(
            pos.evaluate(),
            Evaluator::new(pos.deref().clone()).evaluate()
        )
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_evaluator_is_an_identity(e: Evaluator) {
        assert_eq!(e.to_string().parse(), Ok(e));
    }

    #[rustfmt::skip]
    const SEE_SUITE: &[(&str, &str, i16)] = &[
        ("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", -138),
        ("1k1r4/1ppn3p/p4b2/4n3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", 87),
        ("1n2kb1r/p1P4p/2qb4/5pP1/4n2Q/8/PP1PPP1P/RNB1KBNR w KQk - 0 1", "c7b8q", 122),
        ("1r3r1k/p4pp1/2p1p2p/qpQP3P/2P5/3R4/PP3PP1/1K1R4 b - - 0 1", "a5a2", -585),
        ("1r3r2/5p2/4p2p/2k1n1P1/2PN1nP1/1P3P2/8/2KR1B1R b - - 0 1", "b8b3", -268),
        ("1r3r2/5p2/4p2p/4n1P1/kPPN1nP1/5P2/8/2KR1B1R b - - 0 1", "b8b4", 58),
        ("1r5k/p4pp1/2p1p2p/qpQP3P/2P2P2/1P1R4/P4rP1/1K1R4 b - - 0 1", "a5a2", 57),
        ("2r1k2r/pb4pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w k - 0 1", "d5c6", 0),
        ("2r1k2r/pb4pp/5p1b/2KB3n/4N3/2NP1PB1/PPP1P1PP/R2Q3R w k - 0 1", "d5c6", -210),
        ("2r1k3/pbr3pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w - - 0 1", "d5c6", -178),
        ("2r1r1k1/pp1bppbp/3p1np1/q3P3/2P2P2/1P2B3/P1N1B1PP/2RQ1RK1 b - - 0 1", "d6e5", 56),
        ("2r2r1k/6bp/p7/2q2p1Q/3PpP2/1B6/P5PP/2RR3K b - - 0 1", "c5c1", 50),
        ("2r2rk1/5pp1/pp5p/q2p4/P3n3/1Q3NP1/1P2PP1P/2RR2K1 b - - 0 1", "c8c1", 0),
        ("2r4k/2r4p/p7/2b2p1b/4pP2/1BR5/P1R3PP/2Q4K w - - 0 1", "c3c5", 217),
        ("2r4r/1P4pk/p2p1b1p/7n/BB3p2/2R2p2/P1P2P2/4RK2 w - - 0 1", "c3c8", 326),
        ("3n3r/2P5/8/1k6/8/8/3Q4/4K3 w - - 0 1", "c7d8q", 451),
        ("3N4/2K5/2n5/1k6/8/8/8/8 b - - 0 1", "c6d8", 0),
        ("3q2nk/pb1r1p2/np6/3P2Pp/2p1P3/2R1B2B/PQ3P1P/3R2K1 w - h6 0 1", "g5h6", 57),
        ("3q2nk/pb1r1p2/np6/3P2Pp/2p1P3/2R4B/PQ3P1P/3R2K1 w - h6 0 1", "g5h6", 0),
        ("3r3k/3r4/2n1n3/8/3p4/2PR4/1B1Q4/3R3K w - - 0 1", "d3d4", -131),
        ("4kbnr/p1P1pppp/b7/4q3/7n/8/PP1PPPPP/RNBQKBNR w KQk - 0 1", "c7c8q", -57),
        ("4kbnr/p1P1pppp/b7/4q3/7n/8/PPQPPPPP/RNB1KBNR w KQk - 0 1", "c7c8q", 153),
        ("4kbnr/p1P4p/b1q5/5pP1/4n2Q/8/PP1PPP1P/RNB1KBNR w KQk f6 0 1", "g5f6", 0),
        ("4kbnr/p1P4p/b1q5/5pP1/4n3/5Q2/PP1PPP1P/RNB1KBNR w KQk f6 0 1", "g5f6", 0),
        ("4q3/1p1pr1k1/1B2rp2/6p1/p3PP2/P3R1P1/1P2R1K1/4Q3 b - - 0 1", "e6e4", -268),
        ("4q3/1p1pr1kb/1B2rp2/6p1/p3PP2/P3R1P1/1P2R1K1/4Q3 b - - 0 1", "h7e4", 58),
        ("4r1k1/5pp1/nbp4p/1p2p2q/1P2P1b1/1BP2N1P/1B2QPPK/3R4 b - - 0 1", "g4f3", -27),
        ("4R3/2r3p1/5bk1/1p1r1p1p/p2PR1P1/P1BK1P2/1P6/8 b - - 0 1", "h5g4", 0),
        ("4R3/2r3p1/5bk1/1p1r3p/p2PR1P1/P1BK1P2/1P6/8 b - - 0 1", "h5g4", 0),
        ("5k2/p2P2pp/1b6/1p6/1Nn1P1n1/8/PPP4P/R2QK1NR w KQ - 0 1", "d7d8q", 158),
        ("5k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ - 0 1", "d7d8q", 543),
        ("5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - - 0 1", "d6f4", -21),
        ("5rk1/5pp1/2r4p/5b2/2R5/6Q1/R1P1qPP1/5NK1 b - - 0 1", "f5c2", -161),
        ("6k1/1pp4p/p1pb4/6q1/3P1pRr/2P4P/PP1Br1P1/5RKN w - - 0 1", "f1f4", -49),
        ("6r1/4kq2/b2p1p2/p1pPb3/p1P2B1Q/2P4P/2B1R1P1/6K1 w - - 0 1", "f4e5", 58),
        ("6RR/4bP2/8/8/5r2/3K4/5p2/4k3 w - - 0 1", "f7f8n", 131),
        ("6RR/4bP2/8/8/5r2/3K4/5p2/4k3 w - - 0 1", "f7f8q", 158),
        ("6rr/6pk/p1Qp1b1p/2n5/1B3p2/5p2/P1P2P2/4RK1R w - - 0 1", "e1e8", -326),
        ("7R/4bP2/8/8/1q6/3K4/5p2/4k3 w - - 0 1", "f7f8r", -52),
        ("7R/5P2/8/8/6r1/3K4/5p2/4k3 w - - 0 1", "f7f8b", 157),
        ("7R/5P2/8/8/6r1/3K4/5p2/4k3 w - - 0 1", "f7f8q", 539),
        ("7r/5qpk/2Qp1b1p/1N1r3n/BB3p2/5p2/P1P2P2/4RK1R w - - 0 1", "e1e8", -317),
        ("7r/5qpk/p1Qp1b1p/3r3n/BB3p2/5p2/P1P2P2/4RK1R w - - 0 1", "e1e8", 0),
        ("8/4kp2/2npp3/1Nn5/1p2P1P1/7q/1PP1B3/4KR1r b - - 0 1", "h1f1", 0),
        ("8/4kp2/2npp3/1Nn5/1p2PQP1/7q/1PP1B3/4KR1r b - - 0 1", "h1f1", 0),
        ("8/8/1k6/8/8/2N1N3/4p1K1/3n4 w - - 0 1", "c3d1", 52),
        ("8/8/8/1k6/6b1/4N3/2p3K1/3n4 w - - 0 1", "e3d1", 0),
        ("8/pp6/2pkp3/4bp2/2R3b1/2P5/PP4B1/1K6 w - - 0 1", "g2c6", -165),
        ("r1b1k2r/p4npp/1pp2p1b/7n/1N2N3/3P1PB1/PPP1P1PP/R2QKB1R w KQkq - 0 1", "e4d6", 0),
        ("r1bq1r2/pp1ppkbp/4N1p1/n3P1B1/8/2N5/PPP2PPP/R2QK2R w KQ - 0 1", "e6g7", 32),
        ("r1bq1r2/pp1ppkbp/4N1pB/n3P3/8/2N5/PPP2PPP/R2QK2R w KQ - 0 1", "e6g7", 210),
        ("r1bqk1nr/pppp1ppp/2n5/1B2p3/1b2P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "e1g1", 0),
        ("r1bqkb1r/2pp1ppp/p1n5/1p2p3/3Pn3/1B3N2/PPP2PPP/RNBQ1RK1 b kq - 0 1", "c6d4", 0),
        ("r2n3r/2P1P3/4N3/1k6/8/8/8/4K3 w - - 0 1", "e6d8", 183),
        ("r2q1rk1/1b2bppp/p2p1n2/1ppNp3/3nP3/P2P1N1P/BPP2PP1/R1BQR1K1 w - - 0 1", "d5e7", 26),
        ("r2q1rk1/2p1bppp/p2p1n2/1p2P3/4P1b1/1nP1BN2/PP3PPP/RN1QR1K1 b - - 0 1", "g4f3", -26),
        ("r2qk1nr/pp2ppbp/2b3p1/2p1p3/8/2N2N2/PPPP1PPP/R1BQR1K1 w kq - 0 1", "f3e5", 50),
        ("r2qkbn1/ppp1pp1p/3p1rp1/3Pn3/4P1b1/2N2N2/PPP2PPP/R1BQKB1R b KQq - 0 1", "g4f3", 24),
        ("r4k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ - 0 1", "d7d8q", -57),
        ("r4rk1/1q1nppbp/b2p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - - 0 1", "f6d5", -117),
        ("r4rk1/3nppbp/bq1p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - - 0 1", "b6b2", -600),
        ("rn2k2r/1bq2ppp/p2bpn2/1p1p4/3N4/1BN1P3/PPP2PPP/R1BQR1K1 b kq - 0 1", "d6h2", 50),
        ("rnb1k2r/p3p1pp/1p3p1b/7n/1N2N3/3P1PB1/PPP1P1PP/R2QKB1R w KQkq - 0 1", "e4d6", -117),
        ("rnb2b1r/ppp2kpp/5n2/4P3/q2P3B/5R2/PPP2PPP/RN1QKB2 w Q - 0 1", "h4f6", 24),
        ("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/P1N5/1PQ1PPPP/R1B1KBNR b KQ - 0 1", "b4c3", -26),
        ("rnbqk2r/pp3ppp/2p1pn2/3p4/3P4/N1P1BN2/PPB1PPPb/R2Q1RK1 w kq - 0 1", "g1h2", 193),
        ("rnbqrbn1/pp3ppp/3p4/2p2k2/4p3/3B1K2/PPP2PPP/RNB1Q1NR w - - 0 1", "d3e4", 50),
        ("rnq1k2r/1b3ppp/p2bpn2/1p1p4/3N4/1BN1P3/PPP2PPP/R1BQR1K1 b kq - 0 1", "d6h2", -143),
    ];

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn see_estimates_quiescent_move_gain(
        #[strategy(select(SEE_SUITE))] entry: (&'static str, &'static str, i16),
    ) {
        let (fen, uci, value) = entry;
        let e: Evaluator = fen.parse()?;
        let m = e.moves().unpack().find(|m| m.to_string() == uci).unwrap();
        assert_eq!(e.see(m, Value::lower()..Value::upper()), value);

        assert!(e.winning(m, Value::new(value)));
        assert!(e.winning(m, Value::new(value - 1)));
        assert!(!e.winning(m, Value::new(value + 1)));
    }
}
