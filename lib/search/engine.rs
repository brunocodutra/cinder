use crate::chess::{Move, MoveSet};
use crate::nnue::{Evaluator, Value};
use crate::search::{ControlFlow::*, *};
use crate::{params::Params, syzygy::Syzygy, util::*};
use bytemuck::{Zeroable, fill_zeroes, zeroed};
use derive_more::with_trait::{Constructor, Debug, Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{cell::SyncUnsafeCell, ops::Range, path::Path, pin::Pin, ptr::NonNull, slice};

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

#[derive(Debug)]
struct SharedData {
    syzygy: Syzygy,
    tt: TranspositionTable,
    vt: ValueTable,
}

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
        tpos.best.is_none_or(|m| pos.is_legal(m)).then_some(tpos)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn evaluate(&mut self) -> Value {
        let zobrist = self.stack.pos.zobrists().hash;
        if let Some(value) = self.shared.vt.load(zobrist) {
            return value;
        }

        let value = self.stack.pos.evaluate();
        self.shared.vt.store(zobrist, value);
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
        let pawns = self.local.corrections.pawns.get(pos, zbs.pawns) as f32;
        let minor = self.local.corrections.minor.get(pos, zbs.minor) as f32;
        let major = self.local.corrections.major.get(pos, zbs.major) as f32;
        let white = self.local.corrections.white.get(pos, zbs.white) as f32;
        let black = self.local.corrections.black.get(pos, zbs.black) as f32;

        let mut correction = 0.;
        correction = Params::pawns_correction(0).mul_add(pawns, correction);
        correction = Params::minor_correction(0).mul_add(minor, correction);
        correction = Params::major_correction(0).mul_add(major, correction);
        correction = Params::pieces_correction(0).mul_add(white, correction);
        correction = Params::pieces_correction(0).mul_add(black, correction);
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
        let idx = pos.ply().cast::<usize>();
        let history_bonus = Self::history_bonus(depth).to_int();
        let history_penalty = Self::history_penalty(depth).to_int();
        let continuation_bonus = Self::continuation_bonus(depth).to_int();
        let continuation_penalty = Self::continuation_penalty(depth).to_int();

        self.local.history.update(pos, best, history_bonus);

        for i in 1..=idx.min(2) {
            let reply = self.stack.replies.get_mut(idx - i).assume();
            reply.update(pos, best, continuation_bonus);
        }

        for m in moves.iter() {
            if m == best {
                break;
            }

            self.local.history.update(pos, m, history_penalty);

            for i in 1..=idx.min(2) {
                let reply = self.stack.replies.get_mut(idx - i).assume();
                reply.update(pos, m, continuation_penalty);
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
        self.shared.vt.prefetch(self.stack.pos.zobrists().hash);
        self.shared.tt.prefetch(self.stack.pos.zobrists().hash);

        RecursionGuard::new(self)
    }

    /// The alpha-beta search.
    #[inline(always)]
    fn ab<const IS_PV: bool, const N: usize>(
        &mut self,
        depth: Depth,
        bounds: Range<Score>,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        const { assert!(IS_PV || N == 0) }

        if depth <= 0 {
            Ok(self.quiesce::<IS_PV>(bounds)?.truncate())
        } else if self.stack.pos.ply() >= N as i32 {
            Ok(self.pvs::<IS_PV, 0>(depth, bounds, cut)?.truncate())
        } else {
            self.pvs::<IS_PV, N>(depth, bounds, cut)
        }
    }

    /// The zero-window alpha-beta search.
    #[inline(always)]
    fn nw(&mut self, depth: Depth, beta: Score, cut: bool) -> Result<Pv<0>, Interrupted> {
        if depth > 0 {
            self.pvs::<false, 0>(depth, beta - 1..beta, cut)
        } else {
            self.qnw(beta)
        }
    }

    /// The zero-window quiescent search.
    #[inline(always)]
    fn qnw(&mut self, beta: Score) -> Result<Pv<0>, Interrupted> {
        self.quiesce::<false>(beta - 1..beta)
    }

    /// The quiescent search.
    #[inline(always)]
    fn quiesce<const IS_PV: bool>(&mut self, bounds: Range<Score>) -> Result<Pv<0>, Interrupted> {
        self.stack.nodes.update(1);
        let ply = self.stack.pos.ply();
        if self.ctrl.check(zero(), ply, &self.stack.pv) == Abort {
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

        let transposition = self.transposition();
        let transposed = match transposition {
            None => Pv::empty(self.stack.value[ply.cast::<usize>()].saturate()),
            Some(t) => t.transpose(ply),
        };

        #[expect(clippy::collapsible_if)]
        if !IS_PV && self.stack.pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0) {
            if let Some(t) = transposition {
                let (lower, upper) = t.score.range(ply).into_inner();
                if upper <= alpha || lower >= beta {
                    return Ok(transposed.truncate());
                }
            }
        }

        let alpha = alpha.max(transposed.score());
        if alpha >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        }

        let improving = self.improving();
        let is_check = self.stack.pos.is_check();
        let was_pv = IS_PV || transposition.is_some_and(|t| t.was_pv);
        let mut moves = Moves::from_iter(self.stack.pos.moves().unpack_if(MoveSet::is_noisy));

        moves.sort(|m| {
            if Some(m) == transposed.head() {
                return Bounded::upper();
            }

            let mut rating = 0.;
            let pos = &self.stack.pos;

            let history = self.local.history.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
            rating = Params::history_rating(0).mul_add(history, rating);

            for i in 1..=ply.cast::<usize>().min(2) {
                let reply = self.stack.replies.get_mut(ply.cast::<usize>() - i).assume();
                let history = reply.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
                rating = Params::history_rating(i).mul_add(history, rating);
            }

            if pos.winning(m, Params::good_noisy_margin(0).to_int()) {
                rating += pos.gain(m).to_float::<f32>();
                rating += *Params::good_noisy_bonus(0);
            }

            rating.to_int()
        });

        let mut sorted_moves = moves.sorted();
        let (mut head, mut tail) = match sorted_moves.next() {
            Some(m) => (m, -self.next(Some(m)).quiesce::<IS_PV>(-beta..-alpha)?),
            None => return Ok(transposed.truncate()),
        };

        for (index, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if !IS_PV && !is_check {
                let scale = Params::lmp_improving(0).mul_add(improving, 1.);
                if index.to_float::<f32>() > Params::lmp_scalar(0) * scale {
                    break;
                }
            }

            let pos = &self.stack.pos;
            let mut fut = *Params::fut_margin_scalar(0);
            fut = Params::fut_margin_is_check(0).mul_add(is_check.to_float(), fut);
            fut = Params::fut_margin_gain(0).mul_add(pos.gain(m).to_float(), fut);
            if self.stack.value[ply.cast::<usize>()] + fut.to_int::<i16>().max(0) <= alpha {
                continue;
            }

            if !pos.winning(m, Params::nsp_margin_scalar(0).to_int()) {
                continue;
            }

            let mut next = self.next(Some(m));
            let pv = match -next.qnw(-alpha)? {
                pv if pv <= alpha || pv >= beta => pv,
                _ => -next.quiesce::<IS_PV>(-beta..-alpha)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        let tail = tail.clamp(transposed.score(), Score::upper());
        let score = ScoreBound::new(bounds, tail.score(), ply);
        let tpos = Transposition::new(score, zero(), Some(head), was_pv);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
        Ok(tail.transpose(head))
    }

    /// The principal variation search.
    #[inline(always)]
    fn pvs<const IS_PV: bool, const N: usize>(
        &mut self,
        mut depth: Depth,
        bounds: Range<Score>,
        mut cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        const { assert!(IS_PV || N == 0) }

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

        depth += is_check as i8;
        depth -= transposition.is_none() as i8;

        if depth <= 0 {
            return Ok(self.quiesce::<IS_PV>(bounds)?.truncate());
        }

        #[expect(clippy::collapsible_if)]
        if !IS_PV && self.stack.pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0) {
            if let Some(t) = transposition {
                let (lower, upper) = t.score.range(ply).into_inner();

                #[expect(clippy::collapsible_if)]
                if let Some(margin) = Self::flp(depth - t.depth) {
                    if upper + margin.to_int::<i16>() <= alpha {
                        return Ok(transposed.truncate());
                    }
                }

                #[expect(clippy::collapsible_if)]
                if let Some(margin) = Self::fhp(depth - t.depth) {
                    if cut && lower - margin.to_int::<i16>() >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let was_pv = IS_PV || transposition.is_some_and(|t| t.was_pv);
        let (lower, upper) = match self.shared.syzygy.wdl_after_zeroing(&self.stack.pos) {
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
        };

        let alpha = alpha.max(lower);
        let improving = self.improving();
        let transposed = transposed.clamp(lower, upper);
        if alpha >= beta || upper <= alpha || lower >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        } else if !IS_PV && !is_check {
            #[expect(clippy::collapsible_if)]
            if let Some(margin) = Self::razoring(depth) {
                if self.stack.value[ply.cast::<usize>()] + margin.to_int::<i16>() <= alpha {
                    let pv = self.qnw(beta)?;
                    if pv <= alpha {
                        return Ok(pv.truncate());
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
                    if -self.next(None).nw(d - 1, -beta + 1, !cut)? >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let mut moves = Moves::from_iter(self.stack.pos.moves().unpack());
        let killer = self.stack.killers[ply.cast::<usize>()];

        moves.sort(|m| {
            if Some(m) == transposed.head() {
                return Bounded::upper();
            }

            let pos = &self.stack.pos;
            let mut rating = *Params::killer_rating(0) * killer.contains(m).to_float::<f32>();

            let history = self.local.history.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
            rating = Params::history_rating(0).mul_add(history, rating);

            for i in 1..=ply.cast::<usize>().min(2) {
                let reply = self.stack.replies.get_mut(ply.cast::<usize>() - i).assume();
                let history = reply.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
                rating = Params::history_rating(i).mul_add(history, rating);
            }

            if m.is_noisy() && pos.winning(m, Params::good_noisy_margin(0).to_int()) {
                rating += pos.gain(m).to_float::<f32>();
                rating += *Params::good_noisy_bonus(0);
            }

            rating.to_int()
        });

        if let Some(t) = transposition {
            let p_beta = beta + Self::probcut(depth).to_int::<i16>();
            let p_depth = depth - 3;

            if !was_pv
                && depth >= 6
                && t.depth >= p_depth
                && t.score.lower(ply) >= p_beta
                && t.best.is_none_or(Move::is_noisy)
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
                    let pv = match -next.qnw(-p_beta + 1)? {
                        pv if pv < p_beta => continue,
                        _ => -next.nw(p_depth - 1, -p_beta + 1, !cut)?,
                    };

                    drop(next);
                    if pv >= p_beta {
                        let score = ScoreBound::new(bounds, pv.score(), ply);
                        let tpos = Transposition::new(score, p_depth, Some(m), was_pv);
                        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
                        return Ok(pv.truncate().transpose(m));
                    }
                }
            }
        }

        let mut head = moves.sorted().next().assume();

        let mut tail = {
            let mut extension = 0i8;
            #[expect(clippy::collapsible_if)]
            if let Some(t) = transposition {
                if t.score.lower(ply) >= beta && t.depth >= depth - 3 && depth >= 6 {
                    extension = 2 + head.is_quiet() as i8;
                    let s_depth = (depth - 1) / 2;
                    let s_beta = beta - Self::single(depth).to_int::<i16>();
                    let d_beta = beta - Self::double(depth).to_int::<i16>();
                    let t_beta = beta - Self::triple(depth).to_int::<i16>();
                    for m in moves.sorted().skip(1) {
                        let pv = -self.next(Some(m)).nw(s_depth - 1, -s_beta + 1, !cut)?;
                        if pv >= beta {
                            return Ok(pv.truncate().transpose(m));
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

            let mut next = self.next(Some(head));
            -next.ab::<IS_PV, _>(depth + extension - 1, -beta..-alpha, false)?
        };

        let is_noisy_pv = transposition.is_some_and(|t| {
            t.best.is_some_and(Move::is_noisy) && !matches!(t.score, ScoreBound::Upper(_))
        });

        for (index, m) in moves.sorted().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if !IS_PV && !is_check {
                let scale = Params::lmp_improving(0).mul_add(improving, 1.);
                if index.to_float::<f32>() > Self::lmp(depth) * scale {
                    break;
                }
            }

            let pos = &self.stack.pos;
            let mut lmr = Self::lmr(depth, index);
            let lmr_depth = depth - lmr.to_int::<i8>();
            let history = self.local.history.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
            let reply = self.stack.replies.get_mut(ply.cast::<usize>() - 1).assume();
            let counter = reply.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
            let is_killer = killer.contains(m);

            let mut fut = Self::futility(lmr_depth);
            fut = Params::fut_margin_is_check(0).mul_add(is_check.to_float(), fut);
            fut = Params::fut_margin_is_killer(0).mul_add(is_killer.to_float(), fut);
            fut = Params::fut_margin_gain(0).mul_add(pos.gain(m).to_float(), fut);
            if self.stack.value[ply.cast::<usize>()] + fut.to_int::<i16>().max(0) <= alpha {
                continue;
            }

            let mut spt = if m.is_quiet() {
                Self::qsp(lmr_depth)
            } else {
                Self::nsp(depth)
            };

            spt = Params::qsp_margin_is_killer(0).mul_add(is_killer.to_float(), spt);
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
            lmr = Params::lmr_history(0).mul_add(history, lmr);
            lmr = Params::lmr_counter(0).mul_add(counter, lmr);

            let lmr = lmr.to_int::<i8>().clamp(0, depth.get().max(1) - 1);
            let pv = match -next.nw(depth - lmr - 1, -alpha, !cut)? {
                pv if pv <= alpha || (pv >= beta && lmr < 1) => pv.truncate(),
                _ => -next.ab::<IS_PV, _>(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
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
        if head.is_quiet() && !score.range(ply).contains(&value) {
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
        if self.ctrl.check(depth, zero(), &self.stack.pv) != Continue {
            return Err(Interrupted);
        }

        let correction = self.correction().to_int::<i16>();
        self.stack.value[0] = self.evaluate() + correction;

        moves.sort(|m| {
            if Some(m) == self.stack.pv.head() {
                return Bounded::upper();
            }

            let mut rating = 0.;
            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
            rating = Params::history_rating(0).mul_add(history, rating);

            if m.is_noisy() && pos.winning(m, Params::good_noisy_margin(0).to_int()) {
                rating += pos.gain(m).to_float::<f32>();
                rating += *Params::good_noisy_bonus(0);
            }

            rating.to_int()
        });

        let mut sorted_moves = moves.sorted();
        let mut head = sorted_moves.next().assume();
        self.stack.nodes = self.ctrl.attention(head);

        let mut next = self.next(Some(head));
        let mut tail = -next.ab::<true, _>(depth - 1, -beta..-alpha, false)?;
        drop(next);

        let is_noisy_pv = self.stack.pv.head().is_some_and(Move::is_noisy);
        for (index, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m).to_float::<f32>() / History::LIMIT as f32;
            self.stack.nodes = self.ctrl.attention(m);

            let mut next = self.next(Some(m));
            let gives_check = next.stack.pos.is_check();

            let mut lmr = Self::lmr(depth, index) + *Params::lmr_is_root(0);
            lmr = Params::lmr_gives_check(0).mul_add(gives_check.to_float(), lmr);
            lmr = Params::lmr_is_noisy_pv(0).mul_add(is_noisy_pv.to_float(), lmr);
            lmr = Params::lmr_history(0).mul_add(history, lmr);

            let lmr = lmr.to_int::<i8>().clamp(0, depth.get().max(1) - 1);
            let pv = match -next.nw(depth - lmr - 1, -alpha, false)? {
                pv if pv <= alpha || (pv >= beta && lmr < 1) => pv.truncate(),
                _ => -next.ab::<true, _>(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        let score = ScoreBound::new(bounds, tail.score(), zero());
        let tpos = Transposition::new(score, depth, Some(head), true);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, moves);
            if head.is_quiet() {
                self.stack.killers[0].insert(head);
            }
        }

        let value = self.stack.value[0];
        if head.is_quiet() && !score.range(zero()).contains(&value) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// An implementation of aspiration windows with iterative deepening.
    #[inline(always)]
    fn aw(&mut self, mut moves: Moves) -> impl Iterator<Item = Info> {
        gen move {
            for depth in Depth::iter() {
                let mut reduction = 0.;
                let mut window = *Params::aw_baseline(depth.cast::<usize>().min(5));
                let mut lower = self.stack.pv.score() - window.to_int::<i16>();
                let mut upper = self.stack.pv.score() + window.to_int::<i16>();

                loop {
                    let draft = depth - reduction.to_int::<i8>().min(3);
                    window = window.mul_add(*Params::aw_gamma(0), *Params::aw_delta(0));
                    let Ok(partial) = self.root(&mut moves, draft, lower..upper) else {
                        return;
                    };

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
                            let (time, nodes) = (self.ctrl.elapsed(), self.ctrl.visited());
                            yield Info::new(depth - 1, time, nodes, self.stack.pv.clone());
                        }

                        _ => {
                            self.stack.pv = partial;
                            let (time, nodes) = (self.ctrl.elapsed(), self.ctrl.visited());
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
    pos: &'p Evaluator,
    ctrl: GlobalControl,
    result: Info,
    channel: Option<UnboundedReceiver<Info>>,
    execution: Option<Execution<'e>>,
}

impl<'e, 'p> Search<'e, 'p> {
    fn new(engine: &'e mut Engine, pos: &'p Evaluator, limits: Limits) -> Self {
        Search {
            ctrl: GlobalControl::new(pos, limits),
            result: Pv::empty(Score::lower()).into(),
            channel: None,
            execution: None,
            engine,
            pos,
        }
    }

    /// Aborts the search.
    ///
    /// Returns true if the search had not already been aborted.
    pub fn abort(&self) {
        self.ctrl.abort();
    }

    /// Concludes the search and returns [`Info`] about the best [`Pv`].
    pub fn conclude(self) -> Info {
        self.result.clone()
    }
}

impl Drop for Search<'_, '_> {
    fn drop(&mut self) {
        if let Some(t) = self.execution.take() {
            self.abort();
            drop(t);
        }
    }
}

impl !Unpin for Search<'_, '_> {}

impl FusedStream for Search<'_, '_> {
    #[inline(always)]
    fn is_terminated(&self) -> bool {
        self.channel
            .as_ref()
            .is_some_and(FusedStream::is_terminated)
    }
}

impl Stream for Search<'_, '_> {
    type Item = Info;

    #[inline(always)]
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = unsafe { self.get_unchecked_mut() };
        if let Some(rx) = &mut this.channel {
            let info = match rx.poll_next_unpin(cx) {
                Poll::Ready(Some(info)) => info,
                poll => return poll,
            };

            this.result = info.clone();
            return Poll::Ready(Some(info));
        }

        #[inline(never)]
        #[expect(clippy::deref_addrof)]
        fn bootstrap(search: &mut Search<'_, '_>, cx: &mut Context<'_>) -> Poll<Option<Info>> {
            let (tx, rx) = unbounded();
            search.channel = Some(rx);

            let ctrl: &GlobalControl = unsafe { &*(&raw const search.ctrl) };
            let pos: &Evaluator = unsafe { &*(&raw const *search.pos) };
            let executor: &mut Executor = unsafe { &mut *(&raw mut search.engine.executor) };
            let shared: &SharedData = unsafe { &*(&raw const search.engine.shared) };
            let local: &[SyncUnsafeCell<LocalData>] =
                unsafe { &*(&raw mut *search.engine.local as *const _) };

            let moves = Moves::from_iter(pos.moves().unpack());
            if let Some(pv) = shared.syzygy.best(pos, &moves) {
                search.result = pv.truncate().into();
                return Poll::Ready(Some(search.result.clone()));
            }

            if matches!((moves.len(), &ctrl.limits().clock), (0, _) | (1, Some(_))) {
                let pv = if let Some(m) = moves.iter().next() {
                    Pv::new(Score::drawn(), Line::singular(m))
                } else if pos.is_check() {
                    Pv::empty(Score::mated(zero()))
                } else {
                    Pv::empty(Score::drawn())
                };

                search.result = pv.into();
                return Poll::Ready(Some(search.result.clone()));
            }

            search.execution = Some(executor.execute(move |idx| {
                let local_ctrl = if idx == 0 {
                    LocalControl::active(ctrl)
                } else {
                    LocalControl::passive(ctrl)
                };

                let local = unsafe { &mut *local.get_unchecked(idx).get() };
                let stack = Stack::new(pos.clone(), Pv::empty(Score::drawn()));
                for info in Searcher::new(local_ctrl, shared, local, stack).aw(moves.clone()) {
                    if idx == 0 {
                        tx.unbounded_send(info).assume();
                    }
                }

                if idx == 0 {
                    ctrl.abort();
                }
            }));

            unsafe { Pin::new_unchecked(search).poll_next(cx) }
        }

        bootstrap(this, cx)
    }
}

/// A chess engine.
#[derive(Debug)]
pub struct Engine {
    executor: Executor,
    shared: SharedData,
    local: HugePages<LocalData>,
}

#[cfg(test)]
impl Arbitrary for Engine {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        any::<Options>()
            .prop_map(|o| Engine::with_options(&o))
            .boxed()
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Initializes the engine with the default [`Options`].
    pub fn new() -> Self {
        Self::with_options(&Options::default())
    }

    /// Initializes the engine with the given [`Options`].
    pub fn with_options(options: &Options) -> Self {
        Engine {
            executor: Executor::new(options.threads),
            local: HugePages::zeroed(options.threads.cast()),
            shared: SharedData {
                syzygy: Syzygy::new(&options.syzygy),
                tt: TranspositionTable::new(options.hash),
                vt: ValueTable::new(options.threads),
            },
        }
    }

    /// Resets the hash size.
    pub fn set_hash(&mut self, hash: HashSize) {
        self.shared.tt.resize(hash);
    }

    /// Resets the thread count.
    pub fn set_threads(&mut self, threads: ThreadCount) {
        self.executor = Executor::new(threads);
        self.local.zeroed_in_place(threads.cast());
        self.shared.vt.resize(threads);
    }

    /// Resets the Syzygy path.
    pub fn set_syzygy<I: IntoIterator<Item: AsRef<Path>>>(&mut self, paths: I) {
        self.shared.syzygy = Syzygy::new(paths);
    }

    /// Resets the engine state.
    pub fn reset(&mut self) {
        let local: &[SyncUnsafeCell<LocalData>] = unsafe { &*(&raw mut *self.local as *const _) };

        let vt: &[SyncUnsafeCell<Atomic<Vault<Value, u64>>>] =
            unsafe { &*(&raw mut **self.shared.vt as *const _) };
        let tt: &[SyncUnsafeCell<Atomic<Vault<Transposition, u64>>>] =
            unsafe { &*(&raw mut **self.shared.tt as *const _) };

        let vt_chunk_size = vt.len().div_ceil(local.len());
        let tt_chunk_size = tt.len().div_ceil(local.len());

        self.executor.execute(move |idx| unsafe {
            let offset = idx * vt_chunk_size;
            let len = vt.len().saturating_sub(offset).min(vt_chunk_size);
            let ptr = vt.as_ptr().add(offset) as *mut Atomic<Vault<Value, u64>>;
            fill_zeroes(slice::from_raw_parts_mut(ptr, len));

            let offset = idx * tt_chunk_size;
            let len = tt.len().saturating_sub(offset).min(tt_chunk_size);
            let ptr = tt.as_ptr().add(offset) as *mut Atomic<Vault<Transposition, u64>>;
            fill_zeroes(slice::from_raw_parts_mut(ptr, len));

            *local.get(idx).assume().get() = zeroed();
        });
    }

    /// Initiates a [`Search`].
    pub fn search<'p>(&mut self, pos: &'p Evaluator, limits: Limits) -> Search<'_, 'p> {
        Search::new(self, pos, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Outcome, Position};
    use proptest::sample::Selector;
    use std::fmt::Debug;
    use std::{thread, time::Duration};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_can_be_resized(s: HashSize, t: HashSize) {
        let mut tt = TranspositionTable::new(s);
        tt.resize(t);
        assert_eq!(tt.len(), TranspositionTable::new(t).len());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_can_be_resized(s: ThreadCount, t: ThreadCount) {
        let mut vt = ValueTable::new(s);
        vt.resize(t);
        assert_eq!(vt.len(), ValueTable::new(t).len());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(!#e.shared.tt.is_empty())]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_decisive())] b: Score,
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
        assert_eq!(searcher.nw(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_beta_too_low(
        #[by_ref]
        #[filter(!#e.shared.tt.is_empty())]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_decisive())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_winning() && #s >= #b)] s: Score,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0));

        let tpos = Transposition::new(ScoreBound::Lower(s), Depth::upper(), Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw(d, b, true), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(!#e.shared.tt.is_empty())]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_decisive())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_decisive())] s: Score,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_cut_halfmove_limit(0));

        let tpos = Transposition::new(ScoreBound::Exact(s), Depth::upper(), Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw(d, b, true), Ok(Pv::empty(s)));
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

        assert_eq!(searcher.ab::<true, 1>(d, b, cut), Err(Interrupted));
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

        assert_eq!(searcher.ab::<true, 1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        mut e: Engine,
        #[filter(#pos.outcome().is_some_and(Outcome::is_draw))] pos: Evaluator,
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

        assert_eq!(
            searcher.ab::<true, 1>(d, b, cut),
            Ok(Pv::empty(Score::drawn()))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        mut e: Engine,
        #[filter(#pos.outcome().is_some_and(Outcome::is_decisive))] pos: Evaluator,
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
            searcher.ab::<true, 1>(d, b, cut),
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
        let pos = Evaluator::new(pos);
        let moves = Moves::from_iter(pos.moves().unpack());
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::empty(s));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        let last = searcher.aw(moves).last();
        assert_ne!(last.and_then(|info| info.pv().head()), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn aw_extends_depth_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        s: Score,
    ) {
        let pos = Evaluator::new(pos);
        let moves = Moves::from_iter(pos.moves().unpack());
        let global = GlobalControl::new(&pos, Limits::depth(Depth::lower()));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::empty(s));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        let last = searcher.aw(moves).last();
        assert_ne!(last.and_then(|info| info.pv().head()), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn aw_extends_nodes_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        s: Score,
    ) {
        let pos = Evaluator::new(pos);
        let moves = Moves::from_iter(pos.moves().unpack());
        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::empty(s));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        let last = searcher.aw(moves).last();
        assert_ne!(last.and_then(|info| info.pv().head()), None);
    }
}
