use crate::nnue::{Evaluator, Value};
use crate::search::{ControlFlow::*, *};
use crate::util::{Assume, Atomic, Bits, Bounded, Cache, Float, HugeSeq, Int, Vault};
use crate::{chess::Move, params::Params, syzygy::Syzygy};
use bytemuck::{Zeroable, fill_zeroes, zeroed};
use derive_more::with_trait::{Constructor, Debug, Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{cell::SyncUnsafeCell, io, ops::Range, path::Path, pin::Pin, ptr::NonNull, slice};

#[cfg(test)]
use proptest::prelude::*;

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn convolve<const N: usize>(data: [(f32, &[f32]); N]) -> f32 {
    let mut acc = [0.; N];

    for i in 0..N {
        for j in i..N {
            let param = *data[i].1.get(j - i).assume();
            // The order of operands matters to code gen!
            acc[(i + j) % N] = data[i].0.mul_add(param * data[j].0, acc[(i + j) % N]);
        }
    }

    acc.iter().sum()
}

#[derive(Debug, Display, Copy, Hash, Error)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[display("the search was interrupted")]
struct Interrupted;

#[derive(Debug, Zeroable)]
struct Corrections {
    pawns: Correction,
    minor: Correction,
    major: Correction,
    white: Correction,
    black: Correction,
}

#[derive(Debug, Zeroable)]
struct LocalData {
    history: History,
    continuation: Continuation,
    corrections: Corrections,
}

#[derive(Debug, Deref, DerefMut)]
#[debug("TranspositionTable({})", _0.len())]
struct TranspositionTable(Cache<Transposition>);

impl TranspositionTable {
    #[inline(always)]
    fn new(size: HashSize) -> io::Result<Self> {
        Ok(Self(Cache::new(size.get())?))
    }

    #[inline(always)]
    fn resize(&mut self, size: HashSize) -> io::Result<()> {
        self.0.resize(size.get())
    }
}

#[derive(Debug, Deref, DerefMut)]
#[debug("ValuesTable({})", _0.len())]
struct ValuesTable(Cache<Value, u64>);

impl ValuesTable {
    #[inline(always)]
    fn size(threads: ThreadCount) -> usize {
        (2 + threads.cast::<usize>().next_multiple_of(2)) << 20
    }

    #[inline(always)]
    fn new(threads: ThreadCount) -> io::Result<Self> {
        Ok(Self(Cache::new(Self::size(threads))?))
    }

    #[inline(always)]
    fn resize(&mut self, threads: ThreadCount) -> io::Result<()> {
        self.0.resize(Self::size(threads))
    }
}

#[derive(Debug)]
struct SharedData {
    syzygy: Syzygy,
    tt: TranspositionTable,
    values: ValuesTable,
}

#[derive(Debug)]
struct Stack {
    pv: Pv,
    pos: Evaluator,
    nodes: Option<NonNull<Nodes>>,
    replies: [Option<NonNull<Reply>>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
}

impl Stack {
    #[inline(always)]
    fn new(pos: Evaluator, pv: Pv) -> Self {
        Self {
            pv,
            pos,
            nodes: None,
            replies: [None; Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            value: [Default::default(); Ply::MAX as usize + 1],
        }
    }
}

#[derive(Debug, Constructor, Deref, DerefMut)]
struct RecursionGuard<'e, 'a> {
    searcher: &'e mut Searcher<'a>,
}

impl Drop for RecursionGuard<'_, '_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.searcher.stack.pos.pop();
    }
}

#[derive(Debug, Constructor)]
struct Searcher<'a> {
    ctrl: LocalControl<'a>,
    shared: &'a SharedData,
    local: &'a mut LocalData,
    stack: Stack,
}

impl<'a> Searcher<'a> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn transposition(&self) -> Option<Transposition> {
        let pos = &self.stack.pos;
        let tpos = self.shared.tt.load(pos.zobrists().hash)?;
        tpos.best().is_none_or(|m| pos.is_legal(m)).then_some(tpos)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn evaluate(&mut self) -> Value {
        let zobrist = self.stack.pos.zobrists().hash;
        if let Some(value) = self.shared.values.load(zobrist) {
            return value;
        }

        let value = self.stack.pos.evaluate();
        self.shared.values.store(zobrist, value);
        value
    }

    /// The mate distance pruning.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mdp(&self, bounds: &Range<Score>) -> (Score, Score) {
        let ply = self.stack.pos.ply();
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn correction(&mut self) -> f32 {
        let pos = &self.stack.pos;
        let zbs = pos.zobrists();
        let phase = pos.phase().cast::<usize>();
        let pawns = self.local.corrections.pawns.get(pos, zbs.pawns) as f32;
        let minor = self.local.corrections.minor.get(pos, zbs.minor) as f32;
        let major = self.local.corrections.major.get(pos, zbs.major) as f32;
        let white = self.local.corrections.white.get(pos, zbs.white) as f32;
        let black = self.local.corrections.black.get(pos, zbs.black) as f32;

        let mut correction = 0.;
        correction = Params::pawns_correction(phase).mul_add(pawns, correction);
        correction = Params::minor_correction(phase).mul_add(minor, correction);
        correction = Params::major_correction(phase).mul_add(major, correction);
        correction = Params::pieces_correction(phase).mul_add(white, correction);
        correction = Params::pieces_correction(phase).mul_add(black, correction);
        correction / Correction::LIMIT as f32
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update_correction(&mut self, depth: Depth, score: ScoreBound) {
        let pos = &self.stack.pos;
        let ply = pos.ply();
        let zbs = pos.zobrists();
        let diff = score.bound(ply) - self.stack.value[ply.cast::<usize>()];
        let error = diff.to_float::<f32>() * depth.get().max(1).ilog2().to_float::<f32>();

        let corrections = &mut self.local.corrections;
        let bonus = Params::pawns_correction_bonus(0) * error;
        corrections.pawns.update(pos, zbs.pawns, bonus.to_int());
        let bonus = Params::minor_correction_bonus(0) * error;
        corrections.minor.update(pos, zbs.minor, bonus.to_int());
        let bonus = Params::major_correction_bonus(0) * error;
        corrections.major.update(pos, zbs.major, bonus.to_int());
        let bonus = Params::pieces_correction_bonus(0) * error;
        corrections.white.update(pos, zbs.white, bonus.to_int());
        corrections.black.update(pos, zbs.black, bonus.to_int());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn history_bonus(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::history_bonus_depth(..)),
            (1., Params::history_bonus_scalar(..)),
        ])
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn continuation_bonus(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::continuation_bonus_depth(..)),
            (1., Params::continuation_bonus_scalar(..)),
        ])
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn history_penalty(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::history_penalty_depth(..)),
            (1., Params::history_penalty_scalar(..)),
        ])
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn continuation_penalty(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::continuation_penalty_depth(..)),
            (1., Params::continuation_penalty_scalar(..)),
        ])
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update_history(&mut self, depth: Depth, best: Move, moves: &Moves) {
        let pos = &self.stack.pos;
        let ply = pos.ply();

        let bonus = Self::history_bonus(depth);
        self.local.history.update(pos, best, bonus.to_int());

        let bonus = Self::continuation_bonus(depth);
        let replies = &mut self.stack.replies;
        let mut reply = replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
        reply.update(pos, best, bonus.to_int());

        for m in moves.iter() {
            if m == best {
                break;
            } else {
                let penalty = Self::history_penalty(depth);
                self.local.history.update(pos, m, penalty.to_int());

                let penalty = Self::continuation_penalty(depth);
                let replies = &mut self.stack.replies;
                let mut reply = replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
                reply.update(pos, m, penalty.to_int());
            }
        }
    }

    /// A measure for how much the position is improving.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn improving(&self) -> f32 {
        let pos = &self.stack.pos;
        if pos.is_check() {
            return 0.;
        }

        let ply = pos.ply();
        let idx = ply.cast::<usize>();
        let value = self.stack.value[idx];

        let a = idx >= 2 && !pos[ply - 2].is_check() && value > self.stack.value[idx - 2];
        let b = idx >= 4 && !pos[ply - 4].is_check() && value > self.stack.value[idx - 4];

        let mut idx = Bits::<u8, 2>::new(0);
        idx.push(Bits::<u8, 1>::new(b.cast()));
        idx.push(Bits::<u8, 1>::new(a.cast()));
        *Params::improving(idx.cast::<usize>())
    }

    /// Computes the null move reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nmr(depth: Depth, surplus: Score) -> Option<f32> {
        match depth.get() {
            ..3 => None,
            d @ 3.. => match surplus.get() {
                ..1 => None,
                s @ 1.. => {
                    let gamma = *Params::nmr_gamma(0);
                    let delta = *Params::nmr_delta(0);
                    let limit = *Params::nmr_limit(0);
                    let flat = gamma.mul_add(s.to_float(), delta).min(limit);
                    Some(Params::nmr_fraction(0).mul_add(d.to_float(), flat))
                }
            },
        }
    }

    /// Computes the null move pruning margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nmp(depth: Depth) -> Option<f32> {
        match depth.get() {
            ..1 | 5.. => None,
            d @ 1..5 => Some(convolve([
                (d.to_float(), Params::nmp_margin_depth(..)),
                (1., Params::nmp_margin_scalar(..)),
            ])),
        }
    }

    /// Computes the fail-low pruning reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn flp(depth: Depth) -> Option<f32> {
        match depth.get() {
            5.. => None,
            ..1 => Some(0.),
            d @ 1..5 => Some(convolve([
                (d.to_float(), Params::flp_margin_depth(..)),
                (1., Params::flp_margin_scalar(..)),
            ])),
        }
    }

    /// Computes fail-high pruning reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn fhp(depth: Depth) -> Option<f32> {
        match depth.get() {
            7.. => None,
            ..1 => Some(0.),
            d @ 1..7 => Some(convolve([
                (d.to_float(), Params::fhp_margin_depth(..)),
                (1., Params::fhp_margin_scalar(..)),
            ])),
        }
    }

    /// Computes the razoring margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn razoring(depth: Depth) -> Option<f32> {
        match depth.get() {
            ..1 | 5.. => None,
            d @ 1..5 => Some(convolve([
                (d.to_float(), Params::razoring_depth(..)),
                (1., Params::razoring_scalar(..)),
            ])),
        }
    }

    /// Computes the reverse futility margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn rfp(depth: Depth) -> Option<f32> {
        match depth.get() {
            ..1 | 9.. => None,
            d @ 1..9 => Some(convolve([
                (d.to_float(), Params::rfp_margin_depth(..)),
                (1., Params::rfp_margin_scalar(..)),
            ])),
        }
    }

    /// Computes the futility margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn futility(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::fut_margin_depth(..)),
            (1., Params::fut_margin_scalar(..)),
        ])
    }

    /// Computes the probcut margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn probcut(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::probcut_margin_depth(..)),
            (1., Params::probcut_margin_scalar(..)),
        ])
    }

    /// Computes the singular extension margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn single(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::single_extension_margin_depth(..)),
            (1., Params::single_extension_margin_scalar(..)),
        ])
    }

    /// Computes the double extension margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn double(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::double_extension_margin_depth(..)),
            (1., Params::double_extension_margin_scalar(..)),
        ])
    }

    /// Computes the triple extension margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn triple(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::triple_extension_margin_depth(..)),
            (1., Params::triple_extension_margin_scalar(..)),
        ])
    }

    /// Computes the noisy SEE pruning margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nsp(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::nsp_margin_depth(..)),
            (1., Params::nsp_margin_scalar(..)),
        ])
    }

    /// Computes the quiet SEE pruning margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn qsp(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::qsp_margin_depth(..)),
            (1., Params::qsp_margin_scalar(..)),
        ])
    }

    /// Computes the late move pruning threshold.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn lmp(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::lmp_depth(..)),
            (1., Params::lmp_scalar(..)),
        ])
    }

    /// Computes the late move reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn lmr(depth: Depth, index: usize) -> f32 {
        let log_depth = depth.get().max(1).ilog2();
        let log_index = index.max(1).ilog2();

        convolve([
            (log_depth.to_float(), Params::lmr_depth(..)),
            (log_index.to_float(), Params::lmr_index(..)),
            (1., Params::lmr_scalar(..)),
        ])
    }

    #[must_use]
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn next(&mut self, m: Option<Move>) -> RecursionGuard<'_, 'a> {
        self.stack.replies[self.stack.pos.ply().cast::<usize>()] = m.map(|m| {
            let reply = self.local.continuation.reply(&self.stack.pos, m);
            NonNull::from_mut(reply)
        });

        self.stack.pos.push(m);
        self.shared.values.prefetch(self.stack.pos.zobrists().hash);
        self.shared.tt.prefetch(self.stack.pos.zobrists().hash);

        RecursionGuard::new(self)
    }

    /// The zero-window alpha-beta search.
    #[inline(always)]
    fn nw<const N: usize>(
        &mut self,
        depth: Depth,
        beta: Score,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        self.ab(depth, beta - 1..beta, cut)
    }

    /// The alpha-beta search.
    #[inline(always)]
    fn ab<const N: usize>(
        &mut self,
        depth: Depth,
        bounds: Range<Score>,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        if self.stack.pos.ply() < N as i32 && depth > 0 && bounds.start + 1 < bounds.end {
            self.pvs::<true, N>(depth, bounds, cut)
        } else if bounds.start + 1 < bounds.end {
            Ok(self.pvs::<true, 0>(depth, bounds, cut)?.truncate())
        } else {
            Ok(self.pvs::<false, 0>(depth, bounds, cut)?.truncate())
        }
    }

    /// The principal variation search.
    #[inline(always)]
    fn pvs<const IS_PV: bool, const N: usize>(
        &mut self,
        mut depth: Depth,
        bounds: Range<Score>,
        mut cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        self.stack.nodes.update(1);
        let ply = self.stack.pos.ply();
        if self.ctrl.check(depth, ply, &self.stack.pv) == Abort {
            return Err(Interrupted);
        }

        let (alpha, beta) = match self.stack.pos.outcome() {
            None => self.mdp(&bounds),
            Some(o) if o.is_draw() => return Ok(Pv::empty(Score::drawn())),
            Some(_) => return Ok(Pv::empty(Score::mated(ply))),
        };

        if alpha >= beta {
            return Ok(Pv::empty(alpha));
        }

        let correction = self.correction().to_int::<i16>();
        self.stack.value[ply.cast::<usize>()] = self.evaluate() + correction;

        let is_check = self.stack.pos.is_check();
        let transposition = self.transposition();
        let transposed = match transposition {
            None => Pv::empty(self.stack.value[ply.cast::<usize>()].saturate()),
            Some(t) => t.transpose(ply),
        };

        if depth > 0 {
            depth += is_check as i8;
            depth -= transposition.is_none() as i8;
        }

        let was_pv = IS_PV || transposition.as_ref().is_some_and(Transposition::was_pv);
        let is_noisy_pv = transposition.is_some_and(|t| {
            t.best().is_some_and(|m| !m.is_quiet()) && !matches!(t.score(), ScoreBound::Upper(_))
        });

        #[expect(clippy::collapsible_if)]
        if !IS_PV && self.stack.pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0) {
            if let Some(t) = transposition {
                let (lower, upper) = t.score().range(ply).into_inner();

                #[expect(clippy::collapsible_if)]
                if let Some(margin) = Self::flp(depth - t.depth()) {
                    if upper + margin.to_int::<i16>() <= alpha {
                        return Ok(transposed.truncate());
                    }
                }

                #[expect(clippy::collapsible_if)]
                if let Some(margin) = Self::fhp(depth - t.depth()) {
                    if lower - margin.to_int::<i16>() >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let (lower, upper) = if depth <= 0 {
            (transposed.score(), Score::upper())
        } else {
            match self.shared.syzygy.wdl_after_zeroing(&self.stack.pos) {
                None => (Score::lower(), Score::upper()),
                Some(wdl) => {
                    let bounds = Score::losing(Ply::upper())..Score::winning(Ply::upper());
                    let score = ScoreBound::new(bounds, wdl.to_score(ply), ply);
                    let (lower, upper) = score.range(ply).into_inner();
                    if lower >= upper || upper <= alpha || lower >= beta {
                        let bonus = Params::tb_cut_depth_bonus(0).to_int::<i8>();
                        let tpos = Transposition::new(score, depth + bonus, None, was_pv);
                        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
                        return Ok(tpos.transpose(ply).truncate());
                    }

                    (lower, upper)
                }
            }
        };

        let alpha = alpha.max(lower);
        let improving = self.improving();
        let transposed = transposed.clamp(lower, upper);
        if alpha >= beta || upper <= alpha || lower >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        } else if !IS_PV && !is_check && depth > 0 {
            #[expect(clippy::collapsible_if)]
            if let Some(margin) = Self::razoring(depth) {
                if self.stack.value[ply.cast::<usize>()] + margin.to_int::<i16>() <= alpha {
                    let pv = self.nw(Depth::new(0), beta, cut)?;
                    if pv <= alpha {
                        return Ok(pv);
                    }
                }
            }

            if let Some(mut margin) = Self::rfp(depth) {
                margin = Params::rfp_margin_improving(0).mul_add(improving, margin);
                if transposed.score() - margin.to_int::<i16>() >= beta {
                    return Ok(transposed.truncate());
                }
            }

            let turn = self.stack.pos.turn();
            let pawns = self.stack.pos.pawns(turn);
            if (self.stack.pos.by_color(turn) ^ pawns).len() > 1 {
                #[expect(clippy::collapsible_if)]
                if let Some(margin) = Self::nmp(depth) {
                    if transposed.score() - margin.to_int::<i16>() >= beta {
                        return Ok(transposed.truncate());
                    }
                }

                if let Some(r) = Self::nmr(depth, transposed.score() - beta) {
                    let d = depth - r.to_int::<i8>();
                    if -self.next(None).nw::<0>(d - 1, -beta + 1, !cut)? >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let move_pack = self.stack.pos.moves();
        let mut moves = Moves::from_iter(move_pack.unpack_if(|ms| depth > 0 || !ms.is_quiet()));
        let killer = self.stack.killers[ply.cast::<usize>()];

        moves.sort(|m| {
            if Some(m) == transposed.head() {
                return Bounded::upper();
            }

            let pos = &self.stack.pos;
            let mut rating = *Params::killer_rating(0) * killer.contains(m).to_float::<f32>();

            let history = self.local.history.get(pos, m).to_float::<f32>();
            rating = Params::history_rating(0).mul_add(history / History::LIMIT as f32, rating);

            let replies = &mut self.stack.replies;
            let mut reply = replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let counter = reply.get(pos, m).to_float::<f32>();
            rating = Params::counter_rating(0).mul_add(counter / History::LIMIT as f32, rating);

            if !m.is_quiet() && pos.winning(m, Params::winning_rating_margin(0).to_int()) {
                rating += convolve([
                    (pos.gain(m).to_float(), Params::winning_rating_gain(..)),
                    (1., Params::winning_rating_scalar(..)),
                ]);
            }

            rating.to_int()
        });

        if let Some(t) = transposition {
            let p_beta = beta + Self::probcut(depth).to_int::<i16>();
            let p_depth = depth - 3;

            if !was_pv
                && depth >= 6
                && t.depth() >= p_depth
                && t.score().lower(ply) >= p_beta
                && t.best().is_none_or(|m| !m.is_quiet())
            {
                for m in moves.sorted() {
                    if m.is_quiet() {
                        continue;
                    }

                    let margin = p_beta - self.stack.value[ply.cast::<usize>()];
                    if !self.stack.pos.winning(m, margin.saturate()) {
                        continue;
                    }

                    let mut next = self.next(Some(m));
                    let pv = match -next.nw::<0>(Depth::new(0), -p_beta + 1, !cut)? {
                        pv if pv < p_beta => continue,
                        _ => -next.nw(p_depth - 1, -p_beta + 1, !cut)?,
                    };

                    drop(next);
                    if pv >= p_beta {
                        let score = ScoreBound::new(bounds, pv.score(), ply);
                        let tpos = Transposition::new(score, p_depth, Some(m), was_pv);
                        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
                        return Ok(pv.transpose(m));
                    }
                }
            }
        }

        #[expect(clippy::blocks_in_conditions)]
        let (mut head, mut tail) = match { moves.sorted().next() } {
            None => return Ok(transposed.truncate()),
            Some(m) => {
                let mut extension = 0i8;
                #[expect(clippy::collapsible_if)]
                if let Some(t) = transposition {
                    if t.score().lower(ply) >= beta && t.depth() >= depth - 3 && depth >= 6 {
                        extension = 2 + m.is_quiet() as i8;
                        let s_depth = (depth - 1) / 2;
                        let s_beta = beta - Self::single(depth).to_int::<i16>();
                        let d_beta = beta - Self::double(depth).to_int::<i16>();
                        let t_beta = beta - Self::triple(depth).to_int::<i16>();
                        for m in moves.sorted().skip(1) {
                            let pv = -self.next(Some(m)).nw(s_depth - 1, -s_beta + 1, !cut)?;
                            if pv >= beta {
                                return Ok(pv.transpose(m));
                            } else if pv >= s_beta {
                                cut = true;
                                extension = -1;
                                break;
                            } else if pv >= d_beta {
                                extension = extension.min(1);
                            } else if pv >= t_beta {
                                extension = extension.min(2);
                            }
                        }
                    }
                }

                let mut next = self.next(Some(m));
                (m, -next.ab(depth + extension - 1, -beta..-alpha, false)?)
            }
        };

        let mut is_noisy_node = is_check || is_noisy_pv;
        is_noisy_node |= !head.is_quiet() && tail > alpha;
        for (index, m) in moves.sorted().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut lmp = *Params::lmp_baseline(0);
            lmp = Params::lmp_is_pv(0).mul_add(IS_PV.to_float(), lmp);
            lmp = Params::lmp_was_pv(0).mul_add(was_pv.to_float(), lmp);
            lmp = Params::lmp_is_check(0).mul_add(is_check.to_float(), lmp);
            lmp = Params::lmp_improving(0).mul_add(improving, lmp);
            if index.to_float::<f32>() > Self::lmp(depth) * lmp {
                break;
            }

            let pos = &self.stack.pos;
            let mut lmr = Self::lmr(depth, index);
            let lmr_depth = depth - lmr.to_int::<i8>();
            let history = self.local.history.get(pos, m).to_float::<f32>();
            let replies = &mut self.stack.replies;
            let mut reply = replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let counter = reply.get(pos, m).to_float::<f32>();
            let gain = pos.gain(m).to_float::<f32>();
            let is_killer = killer.contains(m);
            let is_quiet = m.is_quiet();

            let mut fut = Self::futility(lmr_depth);
            fut = Params::fut_margin_is_pv(0).mul_add(IS_PV.to_float(), fut);
            fut = Params::fut_margin_was_pv(0).mul_add(was_pv.to_float(), fut);
            fut = Params::fut_margin_is_check(0).mul_add(is_check.to_float(), fut);
            fut = Params::fut_margin_is_killer(0).mul_add(is_killer.to_float(), fut);
            fut = Params::fut_margin_improving(0).mul_add(improving, fut);
            fut = Params::fut_margin_gain(0).mul_add(gain, fut);
            if self.stack.value[ply.cast::<usize>()] + fut.to_int::<i16>().max(0) <= alpha {
                continue;
            }

            let mut spt = if is_quiet {
                Self::qsp(lmr_depth)
            } else {
                Self::nsp(depth)
            };

            spt = Params::sp_margin_is_killer(0).mul_add(is_killer.to_float(), spt);
            if !pos.winning(m, spt.to_int()) {
                continue;
            }

            let mut next = self.next(Some(m));
            let gives_check = next.stack.pos.is_check();

            lmr += *Params::lmr_baseline(0);
            lmr = Params::lmr_is_pv(0).mul_add(IS_PV.to_float(), lmr);
            lmr = Params::lmr_was_pv(0).mul_add(was_pv.to_float(), lmr);
            lmr = Params::lmr_gives_check(0).mul_add(gives_check.to_float(), lmr);
            lmr = Params::lmr_is_noisy_pv(0).mul_add(is_noisy_pv.to_float(), lmr);
            lmr = Params::lmr_is_killer(0).mul_add(is_killer.to_float(), lmr);
            lmr = Params::lmr_cut(0).mul_add(cut.to_float(), lmr);
            lmr = Params::lmr_improving(0).mul_add(improving, lmr);
            lmr = Params::lmr_history(0).mul_add(history / History::LIMIT as f32, lmr);
            lmr = Params::lmr_counter(0).mul_add(counter / History::LIMIT as f32, lmr);

            let pv = match -next.nw(depth - lmr.to_int::<i8>().max(0) - 1, -alpha, !cut)? {
                pv if pv <= alpha || (pv >= beta && lmr < 1.) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
                is_noisy_node |= !head.is_quiet() && tail > alpha;
            }
        }

        let tail = tail.clamp(lower, upper);
        let score = ScoreBound::new(bounds, tail.score(), ply);
        let tpos = Transposition::new(score, depth, Some(head), was_pv);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, &moves);
            if head.is_quiet() {
                self.stack.killers[ply.cast::<usize>()].insert(head);
            }
        }

        let value = self.stack.value[ply.cast::<usize>()];
        if !is_noisy_node && !score.range(ply).contains(&value) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// The root of the principal variation search.
    #[inline(always)]
    fn root(
        &mut self,
        moves: &mut Moves,
        depth: Depth,
        bounds: Range<Score>,
    ) -> Result<Pv, Interrupted> {
        let (alpha, beta) = (bounds.start, bounds.end);
        if self.ctrl.check(depth, Ply::new(0), &self.stack.pv) != Continue {
            return Err(Interrupted);
        }

        let is_check = self.stack.pos.is_check();
        let is_noisy_pv = self.stack.pv.head().is_some_and(|m| !m.is_quiet());
        let correction = self.correction().to_int::<i16>();
        self.stack.value[0] = self.evaluate() + correction;

        moves.sort(|m| {
            if Some(m) == self.stack.pv.head() {
                return Bounded::upper();
            }

            let mut rating = 0.;
            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m).to_float::<f32>();
            rating = Params::history_rating(0).mul_add(history / History::LIMIT as f32, rating);

            if !m.is_quiet() && pos.winning(m, Params::winning_rating_margin(0).to_int()) {
                rating += convolve([
                    (pos.gain(m).to_float(), Params::winning_rating_gain(..)),
                    (1., Params::winning_rating_scalar(..)),
                ]);
            }

            rating.to_int()
        });

        let mut sorted_moves = moves.sorted();
        let mut head = sorted_moves.next().assume();
        self.stack.nodes = self.ctrl.attention(head);
        let mut tail = -self.next(Some(head)).ab(depth - 1, -beta..-alpha, false)?;

        let mut is_noisy_node = is_check || is_noisy_pv;
        is_noisy_node |= !head.is_quiet() && tail > alpha;
        for (index, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut lmp = *Params::lmp_is_root(0);
            lmp = Params::lmp_is_check(0).mul_add(is_check.to_float(), lmp);
            if index.to_float::<f32>() > Self::lmp(depth) * lmp {
                break;
            }

            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m).to_float::<f32>();
            self.stack.nodes = self.ctrl.attention(m);

            let mut next = self.next(Some(m));
            let gives_check = next.stack.pos.is_check();

            let mut lmr = Self::lmr(depth, index);
            lmr += *Params::lmr_is_root(0);
            lmr = Params::lmr_gives_check(0).mul_add(gives_check.to_float(), lmr);
            lmr = Params::lmr_is_noisy_pv(0).mul_add(is_noisy_pv.to_float(), lmr);
            lmr = Params::lmr_history(0).mul_add(history / History::LIMIT as f32, lmr);

            let pv = match -next.nw(depth - lmr.to_int::<i8>().max(0) - 1, -alpha, false)? {
                pv if pv <= alpha || (pv >= beta && lmr < 1.) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
                is_noisy_node |= !head.is_quiet() && tail > alpha;
            }
        }

        let score = ScoreBound::new(bounds, tail.score(), Ply::new(0));
        let tpos = Transposition::new(score, depth, Some(head), true);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, moves);
            if head.is_quiet() {
                self.stack.killers[0].insert(head);
            }
        }

        let value = self.stack.value[0];
        if !is_noisy_node && !score.range(Ply::new(0)).contains(&value) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// An implementation of aspiration windows with iterative deepening.
    #[inline(always)]
    fn aw(&mut self, mut moves: Moves) -> impl Iterator<Item = Info> {
        #[inline(always)]
        gen move {
            for depth in Depth::iter() {
                let mut reduction = 0.;
                let mut window = *Params::aw_baseline(depth.cast::<usize>().min(7));
                let mut lower = self.stack.pv.score() - window.to_int::<i16>();
                let mut upper = self.stack.pv.score() + window.to_int::<i16>();

                loop {
                    let draft = depth - reduction.to_int::<i8>();
                    window = window.mul_add(*Params::aw_gamma(0), *Params::aw_delta(0));
                    let Ok(partial) = self.root(&mut moves, draft, lower..upper) else {
                        return;
                    };

                    let time = self.ctrl.elapsed();
                    let nodes = self.ctrl.visited();

                    match partial.score() {
                        score if (-lower..Score::upper()).contains(&-score) => {
                            let blend = Params::aw_fail_low_blend(0);
                            upper = blend.lerp(lower.to_float(), upper.to_float()).to_int();
                            lower = score - window.to_int::<i16>();
                            reduction = 0.;
                        }

                        score if (upper..Score::upper()).contains(&score) => {
                            upper = score + window.to_int::<i16>();
                            reduction += *Params::aw_fail_high_reduction(0);
                            self.stack.pv = partial;
                            yield Info::new(depth - 1, time, nodes, self.stack.pv.clone());
                        }

                        _ => {
                            self.stack.pv = partial;
                            break yield Info::new(depth, time, nodes, self.stack.pv.clone());
                        }
                    }
                }
            }
        }
    }
}

/// A handle to an ongoing search.
#[derive(Debug)]
pub struct Search<'e, 'p> {
    engine: &'e mut Engine,
    pos: &'p mut Evaluator,
    pv: Pv,
    ctrl: GlobalControl,
    channel: Option<UnboundedReceiver<Info>>,
    task: Option<Task<'e>>,
}

impl<'e, 'p> Search<'e, 'p> {
    #[inline(always)]
    fn new(engine: &'e mut Engine, pos: &'p mut Evaluator, limits: Limits) -> Self {
        Search {
            pv: Pv::empty(Score::lower()),
            ctrl: GlobalControl::new(pos, limits),
            channel: None,
            task: None,
            engine,
            pos,
        }
    }

    /// Aborts the search.
    ///
    /// Returns true if the search had not already been aborted.
    #[inline(always)]
    pub fn abort(&self) {
        self.ctrl.abort();
    }

    /// The best move found by the search so far.
    #[inline(always)]
    pub fn bestmove(&self) -> Option<Move> {
        self.pv.head()
    }
}

impl Drop for Search<'_, '_> {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(t) = self.task.take() {
            self.abort();
            drop(t);
        }
    }
}

impl FusedStream for Pin<&mut Search<'_, '_>> {
    #[inline(always)]
    fn is_terminated(&self) -> bool {
        self.channel
            .as_ref()
            .is_some_and(FusedStream::is_terminated)
    }
}

impl Stream for Pin<&mut Search<'_, '_>> {
    type Item = Info;

    #[expect(clippy::deref_addrof)]
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(rx) = &mut self.channel {
            let info = match rx.poll_next_unpin(cx) {
                Poll::Ready(Some(info)) => info,
                poll => return poll,
            };

            self.pv = info.pv().clone();
            return Poll::Ready(Some(info));
        }

        let (tx, rx) = unbounded();
        self.channel = Some(rx);

        let ctrl: &GlobalControl = unsafe { &*(&raw const self.ctrl) };
        let pos: &mut Evaluator = unsafe { &mut *(&raw mut *self.pos) };
        let executor: &mut Executor = unsafe { &mut *(&raw mut self.engine.executor) };
        let shared: &SharedData = unsafe { &*(&raw const self.engine.shared) };
        let local: &[SyncUnsafeCell<LocalData>] =
            unsafe { &*(&raw mut *self.engine.local as *const _) };

        let moves = Moves::from_iter(pos.moves().unpack());
        if let Some(pv) = shared.syzygy.best(pos, &moves) {
            self.pv = pv.truncate();
            return Poll::Ready(Some(self.pv.clone().into()));
        }

        if matches!((moves.len(), &ctrl.limits().clock), (0, _) | (1, Some(_))) {
            self.pv = if let Some(m) = moves.iter().next() {
                Pv::new(Score::drawn(), Line::singular(m))
            } else if pos.is_check() {
                Pv::empty(Score::mated(Ply::new(0)))
            } else {
                Pv::empty(Score::drawn())
            };

            return Poll::Ready(Some(self.pv.clone().into()));
        }

        self.task = Some(executor.execute(move |idx| {
            let ctrl = if idx == 0 {
                LocalControl::active(ctrl)
            } else {
                LocalControl::passive(ctrl)
            };

            let local = unsafe { &mut *local.get(idx).assume().get() };
            let stack = Stack::new(pos.clone(), Pv::empty(Score::drawn()));
            for info in Searcher::new(ctrl, shared, local, stack).aw(moves.clone()) {
                if idx == 0 {
                    tx.unbounded_send(info).assume();
                }
            }

            if idx == 0 {
                tx.close_channel();
            }
        }));

        self.poll_next(cx)
    }
}

/// A chess engine.
#[derive(Debug)]
pub struct Engine {
    executor: Executor,
    shared: SharedData,
    local: HugeSeq<LocalData>,
}

#[cfg(test)]
impl Arbitrary for Engine {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        any::<Options>()
            .prop_map(|o| Engine::with_options(&o).unwrap())
            .boxed()
    }
}

impl Engine {
    /// Initializes the engine with the default [`Options`].
    pub fn new() -> io::Result<Self> {
        Self::with_options(&Options::default())
    }

    /// Initializes the engine with the given [`Options`].
    pub fn with_options(options: &Options) -> io::Result<Self> {
        Ok(Engine {
            executor: Executor::new(options.threads)?,
            local: HugeSeq::zeroed(options.threads.cast())?,
            shared: SharedData {
                syzygy: Syzygy::new(&options.syzygy)?,
                tt: TranspositionTable::new(options.hash)?,
                values: ValuesTable::new(options.threads)?,
            },
        })
    }

    /// Resets the hash size.
    pub fn set_hash(&mut self, hash: HashSize) -> io::Result<()> {
        self.shared.tt.resize(hash)
    }

    /// Resets the thread count.
    pub fn set_threads(&mut self, threads: ThreadCount) -> io::Result<()> {
        self.executor = Executor::new(threads)?;
        self.local.zeroed_in_place(threads.cast())?;
        self.shared.values.resize(threads)?;
        Ok(())
    }

    /// Resets the Syzygy path.
    pub fn set_syzygy<I: IntoIterator<Item: AsRef<Path>>>(&mut self, paths: I) -> io::Result<()> {
        self.shared.syzygy = Syzygy::new(paths)?;
        Ok(())
    }

    /// Resets the engine state.
    pub fn reset(&mut self) {
        let values: &[SyncUnsafeCell<Atomic<Vault<Value, u64>>>] =
            unsafe { &*(&raw mut **self.shared.values as *const _) };
        let tt: &[SyncUnsafeCell<Atomic<Vault<Transposition>>>] =
            unsafe { &*(&raw mut **self.shared.tt as *const _) };
        let searchers: &[SyncUnsafeCell<LocalData>] =
            unsafe { &*(&raw mut *self.local as *const _) };

        let values_chunk_size = values.len().div_ceil(searchers.len());
        let tt_chunk_size = tt.len().div_ceil(searchers.len());

        self.executor.execute(move |idx| unsafe {
            let offset = idx * values_chunk_size;
            let len = values.len().saturating_sub(offset).min(values_chunk_size);
            let ptr = values.as_ptr().add(offset) as *mut Atomic<Vault<Value, u64>>;
            fill_zeroes(slice::from_raw_parts_mut(ptr, len));

            let offset = idx * tt_chunk_size;
            let len = tt.len().saturating_sub(offset).min(tt_chunk_size);
            let ptr = tt.as_ptr().add(offset) as *mut Atomic<Vault<Transposition>>;
            fill_zeroes(slice::from_raw_parts_mut(ptr, len));

            *searchers.get(idx).assume().get() = zeroed();
        });
    }

    /// Initiates a [`Search`].
    pub fn search<'p>(&mut self, pos: &'p mut Evaluator, limits: Limits) -> Search<'_, 'p> {
        Search::new(self, pos, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::Position;
    use proptest::sample::Selector;
    use std::fmt::Debug;
    use std::{thread, time::Duration};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_can_be_resized(s: HashSize, t: HashSize) {
        let mut tt = TranspositionTable::new(s)?;
        tt.resize(t)?;
        assert_eq!(tt.len(), TranspositionTable::new(t)?.len());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_can_be_resized(s: ThreadCount, t: ThreadCount) {
        let mut vt = ValuesTable::new(s)?;
        vt.resize(t)?;
        assert_eq!(vt.len(), ValuesTable::new(t)?.len());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_beta_too_low(
        #[by_ref]
        #[filter(#e.shared.tt.len() > 0)]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_winning() && !#b.is_losing())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_winning() && #s >= #b)] s: Score,
        cut: bool,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0));

        let tpos = Transposition::new(ScoreBound::Lower(s), Depth::upper(), Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(#e.shared.tt.len() > 0)]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_winning() && !#b.is_losing())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_losing() && #s < #b)] s: Score,
        cut: bool,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0));

        let tpos = Transposition::new(ScoreBound::Upper(s), Depth::upper(), Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(#e.shared.tt.len() > 0)]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_winning() && !#b.is_losing())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_winning() && !#s.is_losing())] s: Score,
        cut: bool,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0));

        let tpos = Transposition::new(ScoreBound::Exact(s), Depth::upper(), Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ab_aborts_if_time_is_up(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        s: Score,
        cut: bool,
    ) {
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        thread::sleep(Duration::from_millis(1));

        assert_eq!(searcher.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ab_can_be_aborted_upon_request(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        s: Score,
        cut: bool,
    ) {
        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        global.abort();

        assert_eq!(searcher.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        mut e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        s: Score,
        cut: bool,
    ) {
        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);

        assert_eq!(searcher.ab::<1>(d, b, cut), Ok(Pv::empty(Score::drawn())));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        mut e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        s: Score,
        cut: bool,
    ) {
        let ply = pos.ply();
        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);

        assert_eq!(
            searcher.ab::<1>(d, b, cut),
            Ok(Pv::empty(Score::mated(ply)))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn aw_extends_time_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        s: Score,
    ) {
        let pos = Evaluator::new(pos.clone());
        let moves = Moves::from_iter(pos.moves().unpack());
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::empty(s));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        let last = searcher.aw(moves).last();
        assert_ne!(last.and_then(|info| info.head()), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn aw_extends_depth_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        s: Score,
    ) {
        let pos = Evaluator::new(pos.clone());
        let moves = Moves::from_iter(pos.moves().unpack());
        let global = GlobalControl::new(&pos, Limits::depth(Depth::lower()));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::empty(s));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        let last = searcher.aw(moves).last();
        assert_ne!(last.and_then(|info| info.head()), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn aw_extends_nodes_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        s: Score,
    ) {
        let pos = Evaluator::new(pos.clone());
        let moves = Moves::from_iter(pos.moves().unpack());
        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::empty(s));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        let last = searcher.aw(moves).last();
        assert_ne!(last.and_then(|info| info.head()), None);
    }
}
