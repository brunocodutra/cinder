use crate::util::{Assume, Num};
use crate::{chess::*, nnue::*, params::Params, search::Ply, simd::*};
use bytemuck::Zeroable;
use derive_more::with_trait::{Debug, Deref};
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, Index, Range};
use std::{array, mem::replace, str::FromStr};

#[cfg(test)]
use proptest::{prelude::*, sample::*};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref)]
struct Attacks {
    #[deref]
    placement: Placement,
    squares: SquareByIdx,
    roles: RoleByIdx,
    attacks: [Wordboard; Color::LEN],
}

impl Attacks {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(pos: &Position) -> Self {
        let occupied = pos.occupied().cast();

        Attacks {
            placement: *pos.placement(),
            squares: *pos.squares(),
            roles: *pos.roles(),
            attacks: pos.threats().map(|t| t.mask(occupied)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CachedAccumulator {
    accumulator: Accumulator,
    attacks: Attacks,
}

impl Default for CachedAccumulator {
    fn default() -> Self {
        let mut cache = CachedAccumulator {
            accumulator: Default::default(),
            attacks: Default::default(),
        };

        Nnue::transformer().refresh(&mut cache.accumulator);
        cache
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Zeroable)]
#[repr(u8)]
enum Pending {
    Refresh = 0,
    Update,
    None,
}

/// A [`Position`] evaluation stack.
#[derive(Debug, Clone, Eq)]
#[debug("Evaluator({})", self.deref())]
pub struct Evaluator {
    ply: Ply,
    positions: [Position; Ply::LEN],
    accumulator: [[Accumulator; Ply::LEN]; Color::LEN],
    pending: [[Pending; Ply::LEN]; Color::LEN],
    cache: [[CachedAccumulator; KingBucket::LEN]; Color::LEN],
}

impl Default for Evaluator {
    #[inline(always)]
    fn default() -> Self {
        Self::new(Position::default())
    }
}

impl PartialEq for Evaluator {
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

impl Deref for Evaluator {
    type Target = Position;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.index(self.ply)
    }
}

impl Index<Square> for Evaluator {
    type Output = Place;

    #[inline(always)]
    fn index(&self, sq: Square) -> &Self::Output {
        self.deref().index(sq)
    }
}

impl Index<Ply> for Evaluator {
    type Output = Position;

    #[inline(always)]
    fn index(&self, ply: Ply) -> &Self::Output {
        self.index(ply.cast::<usize>())
    }
}

impl Index<usize> for Evaluator {
    type Output = Position;

    #[inline(always)]
    fn index(&self, idx: usize) -> &Self::Output {
        self.positions.get(idx).assume()
    }
}

#[cfg(test)]
impl Arbitrary for Evaluator {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        (any::<Ply>(), any::<Selector>(), any::<Position>())
            .prop_map(|(plies, selector, pos)| {
                let mut pos = Evaluator::new(pos);

                for _ in 0..plies.cast::<usize>() {
                    if pos.outcome().is_none() {
                        pos.push(selector.try_select(pos.moves()));
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
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(pos: Position) -> Self {
        let mut evaluator = Evaluator {
            ply: zeroed(),
            positions: [pos; Ply::LEN],
            accumulator: zeroed(),
            pending: zeroed(),
            cache: Default::default(),
        };

        evaluator.reset();
        evaluator
    }

    /// The current [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn ply(&self) -> Ply {
        self.ply
    }

    /// Estimates the material gain of a move.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn gain(&self, m: Move) -> f32 {
        let mut gain = 0.0;

        if m.is_noisy() {
            if let Some(victim) = self[m.whither()].role() {
                gain += Params::piece_values(victim.cast::<usize>());
            } else if m.is_capture() {
                gain += Params::piece_values(Role::Pawn.cast::<usize>());
            }

            if let Some(promotion) = m.promotion() {
                gain += Params::piece_values(promotion.cast::<usize>());
                gain -= Params::piece_values(Role::Pawn.cast::<usize>());
            }
        }

        gain
    }

    /// Whether this move wins the exchange by at least `margin`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn gaining(&self, m: Move, margin: f32) -> bool {
        self.see(m, margin - 1f32..margin) >= margin
    }

    /// Computes the static exchange evaluation.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn see(&self, m: Move, bounds: Range<f32>) -> f32 {
        let (mut alpha, mut beta) = (bounds.start, bounds.end);
        let mut score = self.gain(m);
        beta = beta.min(score);

        if alpha >= beta {
            return alpha;
        }

        let role = self[m.whence()].role().assume();

        score -= match m.promotion() {
            None => Params::piece_values(role.cast::<usize>()),
            Some(promotion) => Params::piece_values(promotion.cast::<usize>()),
        };

        alpha = alpha.max(score);

        if alpha >= beta {
            return beta;
        }

        let mut exchanges = self.exchanges(m);

        loop {
            let Some((_, captor)) = exchanges.next() else {
                break beta;
            };

            score = -(score + Params::piece_values(captor.cast::<usize>()));
            beta = beta.min(-score);

            if alpha >= beta {
                break alpha;
            }

            let Some((_, captor)) = exchanges.next() else {
                break alpha;
            };

            score = -(score + Params::piece_values(captor.cast::<usize>()));
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

        self.ply += 1;
        self.pending[0][self.ply] = Pending::Update;
        self.pending[1][self.ply] = Pending::Update;
        self.positions[self.ply] = self.positions[self.ply - 1];

        let Some(m) = m else {
            return self.positions[self.ply].pass();
        };

        let turn = self.turn();
        self.positions[self.ply].play(m);
        if self[m.whither()].role() == Some(Role::King) {
            if KingBucket::new(turn, m.whence()) != KingBucket::new(turn, m.whither()) {
                self.pending[turn][self.ply] = Pending::Refresh;
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
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn reset(&mut self) {
        if self.ply > 0 {
            self.positions[0] = self.positions[self.ply];
            self.ply = zeroed();
        }

        for side in Color::iter() {
            self.pending[side][0] = Pending::Refresh;
            self.refresh(side, zeroed());
        }
    }

    /// Evaluates the [`Position`] at the current [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn evaluate(&mut self) -> f32 {
        for side in Color::iter() {
            let mut idx = self.ply.cast::<usize>();

            loop {
                match self.pending[side][idx] {
                    Pending::Refresh => self.refresh(side, idx.convert().assume()),
                    Pending::Update => idx = idx.checked_sub(1).assume(),
                    Pending::None => {
                        break for i in idx + 1..=self.ply.cast::<usize>() {
                            self.update(side, i.convert().assume());
                        };
                    }
                }
            }
        }

        let us = self.turn();
        let them = self.turn().flip();
        Nnue::nn(self.phase()).forward((
            &self.accumulator[us][self.ply],
            &self.accumulator[them][self.ply],
        ))
    }

    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn refresh(&mut self, side: Color, ply: Ply) {
        debug_assert_eq!(self.pending[side][ply], Pending::Refresh);
        self.pending[side][ply] = Pending::None;

        let new = &self.positions[ply];
        let ksq = new.king(side);
        let bucket = KingBucket::new(side, ksq);
        let cache = &mut self.cache[side][bucket];

        if accumulate_ka_in_place(side, ksq, &cache.attacks, new, &mut cache.accumulator).any() {
            let old = replace(&mut cache.attacks, Attacks::new(new));
            accumulate_ti_in_place(side, ksq, &old, &cache.attacks, |sub, add| {
                Nnue::transformer().accumulate_ti_in_place(&mut cache.accumulator, sub, add);
            });

            accumulate_pp_in_place(side, ksq, &old, &cache.attacks, |sub, add| {
                Nnue::transformer().accumulate_pp_in_place(&mut cache.accumulator, sub, add);
            });
        }

        self.accumulator[side][ply] = cache.accumulator;
    }

    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, side: Color, ply: Ply) {
        debug_assert_eq!(self.pending[side][ply], Pending::Update);
        self.pending[side][ply] = Pending::None;

        (ply > 0).assume();
        let new = &self.positions[ply];
        let old = &self.positions[ply - 1];
        let (left, right) = self.accumulator[side].split_at_mut(ply.cast());
        let (src, dst) = (&left[left.len() - 1], &mut right[0]);

        let ksq = new.king(side);
        if accumulate_ka(side, ksq, old, new, src, dst).any() {
            let (old, new) = (Attacks::new(old), Attacks::new(new));
            accumulate_ti_in_place(side, ksq, &old, &new, |sub, add| {
                Nnue::transformer().accumulate_ti_in_place(dst, sub, add);
            });

            accumulate_pp_in_place(side, ksq, &old, &new, |sub, add| {
                Nnue::transformer().accumulate_pp_in_place(dst, sub, add);
            });
        }
    }
}

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn accumulate_ka(
    side: Color,
    ksq: Square,
    old: &Placement,
    new: &Placement,
    src: &Accumulator,
    dst: &mut Accumulator,
) -> M8x64 {
    let diff: M8x64 = old.pieces().simd_ne(new.pieces()).into();

    if !diff.any() {
        *dst = *src;
    } else {
        let kafts_to_sub = KAFeature::lut(side, ksq, old).to_array();
        let mut to_sub = Bitboard::from(diff & old.occupied()).iter();
        (1..=2).contains(&to_sub.len()).assume();

        let kafts_to_add = KAFeature::lut(side, ksq, new).to_array();
        let mut to_add = Bitboard::from(diff & new.occupied()).iter();
        (1..=2).contains(&to_add.len()).assume();

        let sub = array::from_fn(|_| Some(Num::new(kafts_to_sub[to_sub.next()?])));
        let add = array::from_fn(|_| Some(Num::new(kafts_to_add[to_add.next()?])));

        Nnue::transformer().accumulate_ka(src, dst, sub, add);
    }

    diff
}

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn accumulate_ka_in_place(
    side: Color,
    ksq: Square,
    old: &Placement,
    new: &Placement,
    dst: &mut Accumulator,
) -> M8x64 {
    let diff: M8x64 = old.pieces().simd_ne(new.pieces()).into();

    if diff.any() {
        let kafts_to_sub = KAFeature::lut(side, ksq, old).to_array();
        let to_sub = Bitboard::from(diff & old.occupied());

        let kafts_to_add = KAFeature::lut(side, ksq, new).to_array();
        let to_add = Bitboard::from(diff & new.occupied());

        let mut to_sub = to_sub.iter().map(|sq| Num::new(kafts_to_sub[sq]));
        let mut to_add = to_add.iter().map(|sq| Num::new(kafts_to_add[sq]));

        loop {
            let (sub, add) = (to_sub.next(), to_add.next());
            if sub.is_some() || add.is_some() {
                Nnue::transformer().accumulate_ka_in_place(dst, sub, add);
            } else {
                break;
            }
        }
    }

    diff
}

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn accumulate_ti_in_place<F>(side: Color, ksq: Square, old: &Attacks, new: &Attacks, mut acc: F)
where
    F: FnMut(Option<TIFeature>, Option<TIFeature>),
{
    let captured = old.occupied() & new.occupied() & old.pieces().simd_ne(new.pieces());

    for c in Color::iter() {
        let moved = new.squares[c].to_simd().simd_ne(old.squares[c].to_simd());
        let moved = Simd::splat(moved.to_bitmask().cast());

        let promoted = new.roles[c].to_simd().simd_ne(old.roles[c].to_simd());
        let promoted = Simd::splat(promoted.to_bitmask().cast());

        let captured = captured.to_simd().cast::<u16>();
        let diff = new.attacks[c] ^ old.attacks[c] | moved | promoted | captured;

        let indices = old.attacks[c] & diff;
        let nonzero = Bitboard::from(indices.to_simd().simd_ne(zeroed()));
        let mut to_sub = nonzero.iter().flat_map(|wt| {
            let dst = old[wt].piece().assume();
            indices[wt].iter().filter_map(move |idx| {
                let wc = old.squares[c][idx].assume();
                let src = Piece::new(old.roles[c][idx].assume(), c);
                TIFeature::new(side, ksq, src, wc, dst, wt)
            })
        });

        let indices = new.attacks[c] & diff;
        let nonzero = Bitboard::from(indices.to_simd().simd_ne(zeroed()));
        let mut to_add = nonzero.iter().flat_map(|wt| {
            let dst = new[wt].piece().assume();
            indices[wt].iter().filter_map(move |idx| {
                let wc = new.squares[c][idx].assume();
                let src = Piece::new(new.roles[c][idx].assume(), c);
                TIFeature::new(side, ksq, src, wc, dst, wt)
            })
        });

        loop {
            let (sub, add) = (to_sub.next(), to_add.next());
            if sub.is_some() || add.is_some() {
                acc(sub, add);
            } else {
                break;
            }
        }
    }
}

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn accumulate_pp_in_place<F>(side: Color, ksq: Square, old: &Placement, new: &Placement, mut acc: F)
where
    F: FnMut(Option<PPFeature>, Option<PPFeature>),
{
    let old_white_pawns = old.by_piece(Piece::WhitePawn);
    let old_black_pawns = old.by_piece(Piece::BlackPawn);
    let new_white_pawns = new.by_piece(Piece::WhitePawn);
    let new_black_pawns = new.by_piece(Piece::BlackPawn);
    let diff = (old_white_pawns ^ new_white_pawns) | (old_black_pawns ^ new_black_pawns);

    let pfts = PFeature::lut(side, ksq, old).to_array();
    let mut remaining = Bitboard::from(old_white_pawns | old_black_pawns);
    let mut to_sub = remaining.bitand(diff).iter().flat_map(|s| {
        remaining &= !s.bitboard();
        let pft1 = Num::new(pfts[s]);
        let visible = PPFeature::WINDOW[s.file()] & remaining;
        visible.iter().map(move |t| {
            let pft2 = Num::new(pfts[t]);
            PPFeature::new(pft1, pft2)
        })
    });

    let pfts = PFeature::lut(side, ksq, new).to_array();
    let mut remaining = Bitboard::from(new_white_pawns | new_black_pawns);
    let mut to_add = remaining.bitand(diff).iter().flat_map(|s| {
        remaining &= !s.bitboard();
        let pft1 = Num::new(pfts[s]);
        let visible = PPFeature::WINDOW[s.file()] & remaining;
        visible.iter().map(move |t| {
            let pft2 = Num::new(pfts[t]);
            PPFeature::new(pft1, pft2)
        })
    });

    loop {
        let (sub, add) = (to_sub.next(), to_add.next());
        if sub.is_some() || add.is_some() {
            acc(sub, add);
        } else {
            break;
        }
    }
}

impl FromStr for Evaluator {
    type Err = ParsePositionError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s.parse()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn evaluator_updates_accumulator_lazily(
        #[filter(#pos.outcome().is_none())] mut pos: Evaluator,
    ) {
        assert_eq!(
            pos.evaluate().round(),
            Evaluator::new(*pos).evaluate().round()
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_evaluator_is_an_identity(e: Evaluator) {
        assert_eq!(e.to_string().parse(), Ok(e));
    }

    #[rustfmt::skip]
    const SEE_SUITE: &[(&str, &str, f32)] = &[
        ("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", -134.35757),
        ("1k1r4/1ppn3p/p4b2/4n3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", 37.039505),
        ("1n2kb1r/p1P4p/2qb4/5pP1/4n2Q/8/PP1PPP1P/RNB1KBNR w KQk - 0 1", "c7b8q", 134.35754),
        ("1r3r1k/p4pp1/2p1p2p/qpQP3P/2P5/3R4/PP3PP1/1K1R4 b - - 0 1", "a5a2", -581.86115),
        ("1r3r2/5p2/4p2p/2k1n1P1/2PN1nP1/1P3P2/8/2KR1B1R b - - 0 1", "b8b3", -298.60876),
        ("1r3r2/5p2/4p2p/4n1P1/kPPN1nP1/5P2/8/2KR1B1R b - - 0 1", "b8b4", 50.39684),
        ("1r5k/p4pp1/2p1p2p/qpQP3P/2P2P2/1P1R4/P4rP1/1K1R4 b - - 0 1", "a5a2", 50.39684),
        ("2r1k2r/pb4pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w k - 0 1", "d5c6", 0.0),
        ("2r1k2r/pb4pp/5p1b/2KB3n/4N3/2NP1PB1/PPP1P1PP/R2Q3R w k - 0 1", "d5c6", -201.29071),
        ("2r1k3/pbr3pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w - - 0 1", "d5c6", -184.75441),
        ("2r1r1k1/pp1bppbp/3p1np1/q3P3/2P2P2/1P2B3/P1N1B1PP/2RQ1RK1 b - - 0 1", "d6e5", 50.39684),
        ("2r2r1k/6bp/p7/2q2p1Q/3PpP2/1B6/P5PP/2RR3K b - - 0 1", "c5c1", 65.753235),
        ("2r2rk1/5pp1/pp5p/q2p4/P3n3/1Q3NP1/1P2PP1P/2RR2K1 b - - 0 1", "c8c1", 0.0),
        ("2r4k/2r4p/p7/2b2p1b/4pP2/1BR5/P1R3PP/2Q4K w - - 0 1", "c3c5", 201.29071),
        ("2r4r/1P4pk/p2p1b1p/7n/BB3p2/2R2p2/P1P2P2/4RK2 w - - 0 1", "c3c8", 349.0056),
        ("3n3r/2P5/8/1k6/8/8/3Q4/4K3 w - - 0 1", "c7d8q", 483.36316),
        ("3N4/2K5/2n5/1k6/8/8/8/8 b - - 0 1", "c6d8", 0.0),
        ("3q2nk/pb1r1p2/np6/3P2Pp/2p1P3/2R1B2B/PQ3P1P/3R2K1 w - h6 0 1", "g5h6", 50.39684),
        ("3q2nk/pb1r1p2/np6/3P2Pp/2p1P3/2R4B/PQ3P1P/3R2K1 w - h6 0 1", "g5h6", 0.0),
        ("3r3k/3r4/2n1n3/8/3p4/2PR4/1B1Q4/3R3K w - - 0 1", "d3d4", -115.03424),
        ("4kbnr/p1P1pppp/b7/4q3/7n/8/PP1PPPPP/RNBQKBNR w KQk - 0 1", "c7c8q", -50.39685),
        ("4kbnr/p1P1pppp/b7/4q3/7n/8/PPQPPPPP/RNB1KBNR w KQk - 0 1", "c7c8q", 150.89386),
        ("4kbnr/p1P4p/b1q5/5pP1/4n2Q/8/PP1PPP1P/RNB1KBNR w KQk f6 0 1", "g5f6", 0.0),
        ("4kbnr/p1P4p/b1q5/5pP1/4n3/5Q2/PP1PPP1P/RNB1KBNR w KQk f6 0 1", "g5f6", 0.0),
        ("4q3/1p1pr1k1/1B2rp2/6p1/p3PP2/P3R1P1/1P2R1K1/4Q3 b - - 0 1", "e6e4", -298.60876),
        ("4q3/1p1pr1kb/1B2rp2/6p1/p3PP2/P3R1P1/1P2R1K1/4Q3 b - - 0 1", "h7e4", 50.39684),
        ("4r1k1/5pp1/nbp4p/1p2p2q/1P2P1b1/1BP2N1P/1B2QPPK/3R4 b - - 0 1", "g4f3", -16.5363),
        ("4R3/2r3p1/5bk1/1p1r1p1p/p2PR1P1/P1BK1P2/1P6/8 b - - 0 1", "h5g4", -0.0),
        ("4R3/2r3p1/5bk1/1p1r3p/p2PR1P1/P1BK1P2/1P6/8 b - - 0 1", "h5g4", 0.0),
        ("5k2/p2P2pp/1b6/1p6/1Nn1P1n1/8/PPP4P/R2QK1NR w KQ - 0 1", "d7d8q", 150.89386),
        ("5k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ - 0 1", "d7d8q", 581.86115),
        ("5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - - 0 1", "d6f4", -16.5363),
        ("5rk1/5pp1/2r4p/5b2/2R5/6Q1/R1P1qPP1/5NK1 b - - 0 1", "f5c2", -85.140625),
        ("6k1/1pp4p/p1pb4/6q1/3P1pRr/2P4P/PP1Br1P1/5RKN w - - 0 1", "f1f4", -97.318054),
        ("6r1/4kq2/b2p1p2/p1pPb3/p1P2B1Q/2P4P/2B1R1P1/6K1 w - - 0 1", "f4e5", 50.39684),
        ("6RR/4bP2/8/8/5r2/3K4/5p2/4k3 w - - 0 1", "f7f8n", 134.35757),
        ("6RR/4bP2/8/8/5r2/3K4/5p2/4k3 w - - 0 1", "f7f8q", 150.89386),
        ("6rr/6pk/p1Qp1b1p/2n5/1B3p2/5p2/P1P2P2/4RK1R w - - 0 1", "e1e8", -349.0056),
        ("7R/4bP2/8/8/1q6/3K4/5p2/4k3 w - - 0 1", "f7f8r", -50.39685),
        ("7R/5P2/8/8/6r1/3K4/5p2/4k3 w - - 0 1", "f7f8b", 150.89388),
        ("7R/5P2/8/8/6r1/3K4/5p2/4k3 w - - 0 1", "f7f8q", 581.86115),
        ("7r/5qpk/2Qp1b1p/1N1r3n/BB3p2/5p2/P1P2P2/4RK1R w - - 0 1", "e1e8", -349.0056),
        ("7r/5qpk/p1Qp1b1p/3r3n/BB3p2/5p2/P1P2P2/4RK1R w - - 0 1", "e1e8", 0.0),
        ("8/4kp2/2npp3/1Nn5/1p2P1P1/7q/1PP1B3/4KR1r b - - 0 1", "h1f1", 0.0),
        ("8/4kp2/2npp3/1Nn5/1p2PQP1/7q/1PP1B3/4KR1r b - - 0 1", "h1f1", 0.0),
        ("8/8/1k6/8/8/2N1N3/4p1K1/3n4 w - - 0 1", "c3d1", 50.39684),
        ("8/8/8/1k6/6b1/4N3/2p3K1/3n4 w - - 0 1", "e3d1", 0.0),
        ("8/pp6/2pkp3/4bp2/2R3b1/2P5/PP4B1/1K6 w - - 0 1", "g2c6", -150.89388),
        ("r1b1k2r/p4npp/1pp2p1b/7n/1N2N3/3P1PB1/PPP1P1PP/R2QKB1R w KQkq - 0 1", "e4d6", 0.0),
        ("r1bq1r2/pp1ppkbp/4N1p1/n3P1B1/8/2N5/PPP2PPP/R2QK2R w KQ - 0 1", "e6g7", 16.5363),
        ("r1bq1r2/pp1ppkbp/4N1pB/n3P3/8/2N5/PPP2PPP/R2QK2R w KQ - 0 1", "e6g7", 201.29071),
        ("r1bqk1nr/pppp1ppp/2n5/1B2p3/1b2P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "e1g1", 0.0),
        ("r1bqkb1r/2pp1ppp/p1n5/1p2p3/3Pn3/1B3N2/PPP2PPP/RNBQ1RK1 b kq - 0 1", "c6d4", -0.0000038146973),
        ("r2n3r/2P1P3/4N3/1k6/8/8/8/4K3 w - - 0 1", "e6d8", 184.75441),
        ("r2q1rk1/1b2bppp/p2p1n2/1ppNp3/3nP3/P2P1N1P/BPP2PP1/R1BQR1K1 w - - 0 1", "d5e7", 16.5363),
        ("r2q1rk1/2p1bppp/p2p1n2/1p2P3/4P1b1/1nP1BN2/PP3PPP/RN1QR1K1 b - - 0 1", "g4f3", -16.5363),
        ("r2qk1nr/pp2ppbp/2b3p1/2p1p3/8/2N2N2/PPPP1PPP/R1BQR1K1 w kq - 0 1", "f3e5", 50.39684),
        ("r2qkbn1/ppp1pp1p/3p1rp1/3Pn3/4P1b1/2N2N2/PPP2PPP/R1BQKB1R b KQq - 0 1", "g4f3", 33.86054),
        ("r4k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ - 0 1", "d7d8q", -50.39685),
        ("r4rk1/1q1nppbp/b2p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - - 0 1", "f6d5", -134.35757),
        ("r4rk1/3nppbp/bq1p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - - 0 1", "b6b2", -581.86115),
        ("rn2k2r/1bq2ppp/p2bpn2/1p1p4/3N4/1BN1P3/PPP2PPP/R1BQR1K1 b kq - 0 1", "d6h2", 50.39684),
        ("rnb1k2r/p3p1pp/1p3p1b/7n/1N2N3/3P1PB1/PPP1P1PP/R2QKB1R w KQkq - 0 1", "e4d6", -134.35757),
        ("rnb2b1r/ppp2kpp/5n2/4P3/q2P3B/5R2/PPP2PPP/RN1QKB2 w Q - 0 1", "h4f6", 33.86054),
        ("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/P1N5/1PQ1PPPP/R1B1KBNR b KQ - 0 1", "b4c3", -16.5363),
        ("rnbqk2r/pp3ppp/2p1pn2/3p4/3P4/N1P1BN2/PPB1PPPb/R2Q1RK1 w kq - 0 1", "g1h2", 201.29071),
        ("rnbqrbn1/pp3ppp/3p4/2p2k2/4p3/3B1K2/PPP2PPP/RNB1Q1NR w - - 0 1", "d3e4", 50.39684),
        ("rnq1k2r/1b3ppp/p2bpn2/1p1p4/3N4/1BN1P3/PPP2PPP/R1BQR1K1 b kq - 0 1", "d6h2", -150.89388),
    ];

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn see_estimates_quiescent_move_gain(
        #[strategy(select(SEE_SUITE))] entry: (&'static str, &'static str, f32),
    ) {
        let (fen, uci, value) = entry;
        let e: Evaluator = fen.parse()?;
        let mut moves = e.moves().into_iter();
        let m = moves.find(|m| m.to_string() == uci).unwrap();
        assert_eq!(e.see(m, -f32::MAX..f32::MAX), value);

        assert!(e.gaining(m, value));
        assert!(e.gaining(m, value - 1.0));
        assert!(!e.gaining(m, value + 1.0));
    }
}
