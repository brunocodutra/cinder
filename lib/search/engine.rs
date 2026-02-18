use crate::chess::{Move, MoveSet, Role};
use crate::search::{ControlFlow::*, *};
use crate::{nnue::Evaluator, params::Params, syzygy::Syzygy, util::*};
use bytemuck::{Zeroable, fill_zeroes, zeroed};
use derive_more::with_trait::{Constructor, Debug, Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::ops::{Mul, Range};
use std::task::{Context, Poll};
use std::{cell::SyncUnsafeCell, path::Path, pin::Pin, ptr::NonNull, slice};

#[cfg(test)]
use proptest::prelude::*;

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn convolve<const N: usize>(data: [(f32, &[f32]); N]) -> f32 {
    let mut acc = [0.0; N];

    for i in 0..N {
        for j in i..N {
            let param = *data[i].1.get(j - i).assume();
            acc[(i + j) % N] = param.mul_add(data[i].0 * data[j].0, acc[(i + j) % N]);
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
    pawns: Correction<16384>,
    minor: Correction<16384>,
    major: Correction<16384>,
    white: Correction<16384>,
    black: Correction<16384>,
    history: HistoryCorrection,
}

#[derive(Debug, Zeroable)]
struct LocalData {
    history: History,
    continuation: ContinuationHistory,
    corrections: Corrections,
}

#[derive(Debug)]
struct Stack {
    pv: Pv,
    pos: Evaluator,
    nodes: Option<NonNull<Nodes>>,
    replies: [Option<NonNull<ContinuationHistoryReply>>; Ply::MAX as usize + 1],
    correction: [Option<NonNull<Correction<1>>>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    values: [Value; Ply::MAX as usize + 1],
}

impl Stack {
    #[inline(always)]
    fn new(pos: Evaluator, pv: Pv) -> Self {
        Self {
            pv,
            pos,
            nodes: None,
            replies: [None; Ply::MAX as usize + 1],
            correction: [None; Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            values: [Default::default(); Ply::MAX as usize + 1],
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn value(&self, i: usize) -> Score {
        let idx = self.pos.ply().cast::<usize>().wrapping_sub(i);
        // IMPORTANT: widen to Score to avoid mate blindness!
        self.values.get(idx).assume().saturate()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn reply(&self, i: usize) -> Option<NonNull<ContinuationHistoryReply>> {
        let idx = self.pos.ply().cast::<usize>().wrapping_sub(i);
        *self.replies.get(idx)?
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn correction(&self, i: usize) -> Option<NonNull<Correction<1>>> {
        let idx = self.pos.ply().cast::<usize>().wrapping_sub(i);
        *self.correction.get(idx)?
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
    fn evaluate(&mut self) -> Value {
        let zobrist = self.stack.pos.zobrists().hash;
        self.shared.vt.load(zobrist).unwrap_or_else(|| {
            let value = self.stack.pos.evaluate().saturate();
            self.shared.vt.store(zobrist, value);
            value
        })
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update_history(&mut self, depth: f32, best: Move, moves: &Moves) {
        let history_bonus = Params::history_bonus(0)
            .mul_add(depth, *Params::history_bonus(1))
            .min(*Params::history_bonus(2));

        let history_penalty = Params::history_penalty(0)
            .mul_add(depth, *Params::history_penalty(1))
            .max(*Params::history_penalty(2));

        let counter_bonus = Params::counter_bonus(0)
            .mul_add(depth, *Params::counter_bonus(1))
            .min(*Params::counter_bonus(2));

        let counter_penalty = Params::counter_penalty(0)
            .mul_add(depth, *Params::counter_penalty(1))
            .max(*Params::counter_penalty(2));

        let followup_bonus = Params::followup_bonus(0)
            .mul_add(depth, *Params::followup_bonus(1))
            .min(*Params::followup_bonus(2));

        let followup_penalty = Params::followup_penalty(0)
            .mul_add(depth, *Params::followup_penalty(1))
            .max(*Params::followup_penalty(2));

        let pos = &self.stack.pos;
        let bonus = [history_bonus, counter_bonus, followup_bonus];
        let penalty = [history_penalty, counter_penalty, followup_penalty];

        self.local.history.update(pos, best, bonus[0]);
        for i in 1..=bonus[1..].len().min(pos.ply().cast()) {
            self.stack.reply(i).update(pos, best, bonus[i]);
        }

        for m in moves.iter().take_while(|&m| m != best) {
            self.local.history.update(pos, m, penalty[0]);
            for i in 1..=penalty[1..].len().min(pos.ply().cast()) {
                self.stack.reply(i).update(pos, m, penalty[i]);
            }
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update_correction(&mut self, depth: f32, score: ScoreBound) {
        let pos = &self.stack.pos;
        let (ply, zbs) = (pos.ply(), pos.zobrists());
        let diff = score.bound(ply) - self.stack.value(0);
        let error = depth * diff.cast::<f32>();

        let pawns_delta = error
            .mul(*Params::pawns_correction_delta(0))
            .max(*Params::pawns_correction_delta(1))
            .min(*Params::pawns_correction_delta(2));

        let minor_delta = error
            .mul(*Params::minor_correction_delta(0))
            .max(*Params::minor_correction_delta(1))
            .min(*Params::minor_correction_delta(2));

        let major_delta = error
            .mul(*Params::major_correction_delta(0))
            .max(*Params::major_correction_delta(1))
            .min(*Params::major_correction_delta(2));

        let pieces_delta = error
            .mul(*Params::pieces_correction_delta(0))
            .max(*Params::pieces_correction_delta(1))
            .min(*Params::pieces_correction_delta(2));

        let counter_delta = error
            .mul(*Params::counter_correction_delta(0))
            .max(*Params::counter_correction_delta(1))
            .min(*Params::counter_correction_delta(2));

        let followup_delta = error
            .mul(*Params::followup_correction_delta(0))
            .max(*Params::followup_correction_delta(1))
            .min(*Params::followup_correction_delta(2));

        let corrections = &mut self.local.corrections;
        corrections.pawns.update(pos, zbs.pawns, pawns_delta);
        corrections.minor.update(pos, zbs.minor, minor_delta);
        corrections.major.update(pos, zbs.major, major_delta);
        corrections.white.update(pos, zbs.white, pieces_delta);
        corrections.black.update(pos, zbs.black, pieces_delta);

        let history_deltas = [counter_delta, followup_delta];
        for i in 1..=history_deltas.len().min(ply.cast()) {
            let delta = history_deltas[i - 1];
            self.stack.correction(i).update(pos, (), delta);
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn correction(&self) -> f32 {
        let pos = &self.stack.pos;
        let (ply, zbs) = (pos.ply(), pos.zobrists());

        let mut correction = 0.0;
        let pawns = self.local.corrections.pawns.get(pos, zbs.pawns);
        correction = Params::pawns_correction(0).mul_add(pawns, correction);
        let minor = self.local.corrections.minor.get(pos, zbs.minor);
        correction = Params::minor_correction(0).mul_add(minor, correction);
        let major = self.local.corrections.major.get(pos, zbs.major);
        correction = Params::major_correction(0).mul_add(major, correction);
        let white = self.local.corrections.white.get(pos, zbs.white);
        correction = Params::pieces_correction(0).mul_add(white, correction);
        let black = self.local.corrections.black.get(pos, zbs.black);
        correction = Params::pieces_correction(0).mul_add(black, correction);

        for i in 1..=Params::continuation_correction(..).len().min(ply.cast()) {
            let history = self.stack.correction(i).get(pos, ());
            correction = Params::continuation_correction(i - 1).mul_add(history, correction);
        }

        correction
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn transposition(&self) -> Option<Transposition> {
        let pos = &self.stack.pos;
        let tpos = self.shared.tt.load(pos.zobrists().hash)?;
        tpos.best.is_none_or(|m| pos.is_legal(m)).then_some(tpos)
    }

    /// A measure for how much the position is improving.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn improving(&self) -> f32 {
        let pos = &self.stack.pos;
        let ply = pos.ply();
        if pos.is_check() {
            return 0.0;
        }

        let a = ply >= 2 && !pos[ply - 2].is_check() && self.stack.value(0) > self.stack.value(2);
        let b = ply >= 4 && !pos[ply - 4].is_check() && self.stack.value(0) > self.stack.value(4);

        let mut idx = Bits::<u8, 2>::new(0);
        idx.push(Bits::<u8, 1>::new(b.cast()));
        idx.push(Bits::<u8, 1>::new(a.cast()));
        *Params::improving(idx.cast::<usize>())
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

    /// Computes the null move reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nmr(depth: f32, surplus: Score) -> Option<f32> {
        if depth < *Params::nmr_depth_limit(0) {
            return None;
        }

        match surplus.get() {
            ..1 => None,
            s @ 1.. => {
                let gamma = *Params::nmr_score(0);
                let delta = *Params::nmr_score(1);
                let limit = *Params::nmr_score(2);
                let flat = gamma.mul_add(s.cast(), delta).min(limit);
                Some(Params::nmr_depth(0).mul_add(depth, flat))
            }
        }
    }

    /// Computes the null move pruning margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nmp(depth: f32) -> Option<f32> {
        if depth >= *Params::nmp_depth_limit(0) {
            return None;
        }

        Some(convolve([
            (depth, Params::nmp_margin_depth(..)),
            (1.0, Params::nmp_margin_scalar(..)),
        ]))
    }

    /// Computes fail-high pruning reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn tt_fh(depth: f32) -> f32 {
        match depth {
            ..0.0 => 0.0,
            d => convolve([
                (d, Params::tt_fh_margin_depth(..)),
                (1.0, Params::tt_fh_margin_scalar(..)),
            ]),
        }
    }

    /// Computes the fail-low pruning reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn tt_fl(depth: f32) -> f32 {
        match depth {
            ..0.0 => 0.0,
            d => convolve([
                (d, Params::tt_fl_margin_depth(..)),
                (1.0, Params::tt_fl_margin_scalar(..)),
            ]),
        }
    }

    /// Computes the razoring margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn razoring(depth: f32) -> f32 {
        convolve([
            (depth, Params::razoring_depth(..)),
            (1.0, Params::razoring_scalar(..)),
        ])
    }

    /// Computes the reverse futility margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn rfp(depth: f32) -> f32 {
        convolve([
            (depth, Params::rfp_margin_depth(..)),
            (1.0, Params::rfp_margin_scalar(..)),
        ])
    }

    /// Computes the probcut margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn probcut(depth: f32) -> f32 {
        convolve([
            (depth, Params::probcut_margin_depth(..)),
            (1.0, Params::probcut_margin_scalar(..)),
        ])
    }

    /// Computes the singular extension margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn singular(depth: f32) -> f32 {
        convolve([
            (depth, Params::singular_margin_depth(..)),
            (1.0, Params::singular_margin_scalar(..)),
        ])
    }

    /// Computes the late move pruning threshold.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn lmp(depth: f32) -> f32 {
        convolve([
            (depth, Params::lmp_depth(..)),
            (1.0, Params::lmp_scalar(..)),
        ])
    }

    /// Computes the futility margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn futility(depth: f32) -> f32 {
        convolve([
            (depth, Params::futility_margin_depth(..)),
            (1.0, Params::futility_margin_scalar(..)),
        ])
    }

    /// Computes the SEE pruning margin.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn see_pruning(depth: f32, m: Move) -> f32 {
        if m.is_quiet() {
            convolve([
                (depth, Params::see_margin_quiet_depth(..)),
                (1.0, Params::see_margin_quiet_scalar(..)),
            ])
        } else {
            convolve([
                (depth, Params::see_margin_noisy_depth(..)),
                (1.0, Params::see_margin_noisy_scalar(..)),
            ])
        }
    }

    /// Computes the late move reduction.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn lmr(depth: f32, index: usize) -> f32 {
        convolve([
            (index.max(1).cast::<f32>().ln(), Params::lmr_index(..)),
            (depth.ln(), Params::lmr_depth(..)),
            (1.0, Params::lmr_scalar(..)),
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

        self.stack.correction[self.stack.pos.ply().cast::<usize>()] = m.map(|m| {
            let reply = self.local.corrections.history.get(&self.stack.pos, m);
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
        depth: f32,
        bounds: Range<Score>,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        const { assert!(IS_PV || N == 0) }

        if depth < 1.0 {
            Ok(self.quiesce::<IS_PV>(bounds)?.truncate())
        } else if self.stack.pos.ply() >= N as i32 {
            Ok(self.pvs::<IS_PV, 0>(depth, bounds, cut)?.truncate())
        } else {
            self.pvs::<IS_PV, N>(depth, bounds, cut)
        }
    }

    /// The zero-window alpha-beta search.
    #[inline(always)]
    fn nw(&mut self, depth: f32, beta: Score, cut: bool) -> Result<Pv<0>, Interrupted> {
        if depth >= 1.0 {
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

        let correction = self.correction().cast::<i16>();
        self.stack.values[ply.cast::<usize>()] = self.evaluate() + correction;

        let transposition = self.transposition();
        let transposed = match transposition {
            None => Pv::empty(self.stack.value(0).saturate()),
            Some(t) => t.transpose(ply),
        };

        if !IS_PV && self.stack.pos.halfmoves() as f32 <= *Params::tt_hm_limit(0) {
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

            let mut rating = 0.0;
            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m);
            rating = Params::history_rating(0).mul_add(history, rating);
            if pos.gaining(m, *Params::good_noisy_margin(0)) {
                rating += *Params::good_noisy_rating(0);
                rating += pos.gain(m);
            }

            rating.saturate()
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

            if !IS_PV && !is_check && !tail.is_losing() {
                let scale = Params::lmp_improving(0).mul_add(improving, 1.0);
                if index.cast::<f32>() > Params::lmp_scalar(0) * scale {
                    break;
                }
            }

            let pos = &self.stack.pos;
            if !is_check && !tail.is_losing() {
                let margin = pos.gain(m) + *Params::futility_margin_quiescence(0);
                if self.stack.value(0) + margin.cast::<i16>() <= alpha {
                    continue;
                }
            }

            if !tail.is_losing() && !pos.gaining(m, *Params::see_margin_quiescence(0)) {
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

        let tail = tail.clip(transposed.score(), Score::upper());
        let score = ScoreBound::new(bounds, tail.score(), ply);
        let tpos = Transposition::new(score, zero(), Some(head), was_pv);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
        Ok(tail.transpose(head))
    }

    /// The principal variation search.
    #[inline(always)]
    fn pvs<const IS_PV: bool, const N: usize>(
        &mut self,
        mut depth: f32,
        bounds: Range<Score>,
        mut cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        const { assert!(IS_PV || N == 0) }

        self.stack.nodes.update(1);
        let ply = self.stack.pos.ply();
        if self.ctrl.check(depth.saturate(), ply, &self.stack.pv) == Abort {
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

        let correction = self.correction().cast::<i16>();
        self.stack.values[ply.cast::<usize>()] = self.evaluate() + correction;

        let is_check = self.stack.pos.is_check();
        let transposition = self.transposition();
        let transposed = match transposition {
            None => Pv::empty(self.stack.value(0).saturate()),
            Some(t) => t.transpose(ply),
        };

        depth += is_check.cast::<f32>();
        depth -= transposition.is_none().cast::<f32>();

        if depth < 1.0 {
            return Ok(self.quiesce::<IS_PV>(bounds)?.truncate());
        }

        if !IS_PV && self.stack.pos.halfmoves() as f32 <= *Params::tt_hm_limit(0) {
            if let Some(t) = transposition {
                let tt_depth = t.depth.cast::<f32>();
                let (lower, upper) = t.score.range(ply).into_inner();

                if cut && lower - Self::tt_fh(depth - tt_depth).cast::<i16>() >= beta {
                    return Ok(transposed.truncate());
                }

                if upper + Self::tt_fl(depth - tt_depth).cast::<i16>() <= alpha {
                    return Ok(transposed.truncate());
                }
            }
        }

        let was_pv = IS_PV || transposition.is_some_and(|t| t.was_pv);
        let is_noisy_node = transposition.is_some_and(|t| {
            t.best.is_some_and(Move::is_noisy) && !matches!(t.score, ScoreBound::Upper(_))
        });

        let (lower, upper) = match self.shared.syzygy.wdl_after_zeroing(&self.stack.pos) {
            None => (Score::lower(), Score::upper()),
            Some(wdl) => {
                let bounds = Score::losing(Ply::upper())..Score::winning(Ply::upper());
                let score = ScoreBound::new(bounds, wdl.to_score(ply), ply);
                let (lower, upper) = score.range(ply).into_inner();
                if lower >= upper || upper <= alpha || lower >= beta {
                    let tt_depth = depth + Params::tb_depth_bonus(0);
                    let tpos = Transposition::new(score, tt_depth.saturate(), None, was_pv);
                    self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
                    return Ok(tpos.transpose(ply).truncate());
                }

                (lower, upper)
            }
        };

        let alpha = alpha.max(lower);
        let improving = self.improving();
        let transposed = transposed.clip(lower, upper);
        if alpha >= beta || upper <= alpha || lower >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        } else if !IS_PV && !is_check {
            if alpha < Params::razoring_alpha_limit(0).cast::<i16>() {
                let margin = Self::razoring(depth);
                if self.stack.value(0) + margin.cast::<i16>() <= alpha {
                    let pv = self.qnw(beta)?;
                    if pv <= alpha {
                        return Ok(pv.truncate());
                    }
                }
            }

            if !beta.is_losing() {
                let mut margin = Self::rfp(depth);
                margin = Params::rfp_margin_improving(0).mul_add(improving, margin);
                if self.stack.value(0) - margin.cast::<i16>() >= beta {
                    return Ok(Pv::empty(self.stack.value(0)));
                }
            }

            if !beta.is_losing() && !transposed.is_winning() {
                let turn = self.stack.pos.turn();
                let ours = self.stack.pos.by_color(turn);
                let pawns = self.stack.pos.by_role(Role::Pawn);
                let kings = self.stack.pos.by_role(Role::King);
                if ours & !(pawns ^ kings) != zero() {
                    if let Some(margin) = Self::nmp(depth) {
                        if transposed.score() - margin.cast::<i16>() >= beta {
                            return Ok(transposed.truncate());
                        }
                    }

                    if let Some(r) = Self::nmr(depth, transposed.score() - beta) {
                        if -self.next(None).nw(depth - r - 1.0, -beta + 1, false)? >= beta {
                            return Ok(transposed.truncate());
                        }
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

            let mut rating = *Params::killer_rating(0) * killer.contains(m).cast::<f32>();

            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m);
            rating = Params::history_rating(0).mul_add(history, rating);

            for i in 1..=Params::history_rating(1..).len().min(ply.cast()) {
                let history = self.stack.reply(i).get(pos, m);
                rating = Params::history_rating(i).mul_add(history, rating);
            }

            if m.is_noisy() && pos.gaining(m, *Params::good_noisy_margin(0)) {
                rating += *Params::good_noisy_rating(0);
                rating += pos.gain(m);
            }

            rating.saturate()
        });

        if let Some(t) = transposition {
            let gamma = *Params::probcut_depth(0);
            let delta = *Params::probcut_depth(1);
            let pc_depth = gamma.mul_add(depth, delta);

            let mut margin = Self::probcut(depth);
            margin = Params::probcut_margin_improving(0).mul_add(improving, margin);
            let pc_beta = beta + margin.cast::<i16>();

            let max_depth = t.depth.cast::<f32>() + *Params::probcut_depth_bounds(1);
            let depth_bounds = *Params::probcut_depth_bounds(0)..max_depth;
            if is_noisy_node && t.score.lower(ply) >= pc_beta && depth_bounds.contains(&depth) {
                for m in moves.sorted() {
                    let margin = pc_beta - self.stack.value(0);
                    if m.is_quiet() || !self.stack.pos.gaining(m, margin.cast()) {
                        continue;
                    }

                    let mut next = self.next(Some(m));
                    let pv = match -next.qnw(-pc_beta + 1)? {
                        pv if pv < pc_beta => continue,
                        _ => -next.nw(pc_depth - 1.0, -pc_beta + 1, false)?,
                    };

                    drop(next);
                    if pv >= pc_beta {
                        let score = ScoreBound::new(bounds, pv.score(), ply);
                        let tpos = Transposition::new(score, pc_depth.saturate(), Some(m), was_pv);
                        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);
                        return Ok(pv.truncate().transpose(m));
                    }
                }
            }
        }

        let mut head = moves.sorted().next().assume();

        let mut tail = {
            let mut extension = 0f32;
            if let Some(t) = transposition {
                let max_depth = t.depth.cast::<f32>() + *Params::singular_depth_bounds(1);
                let depth_bounds = *Params::singular_depth_bounds(0)..max_depth;
                if !matches!(t.score, ScoreBound::Upper(_)) && depth_bounds.contains(&depth) {
                    let single = Self::singular(depth);
                    let double = single + Params::singular_margin_scalar(1);
                    let triple = double + Params::singular_margin_scalar(2);
                    let expected_cut = cut || t.score.lower(ply) >= beta;

                    let gamma = *Params::singular_depth(0);
                    let delta = *Params::singular_depth(1);
                    let se_depth = gamma.mul_add(depth, delta);

                    let se_beta = t.score.bound(ply) - single.cast::<i16>();
                    let de_beta = t.score.bound(ply) - double.cast::<i16>();
                    let te_beta = t.score.bound(ply) - triple.cast::<i16>();

                    if expected_cut {
                        extension = 2.0 + head.is_quiet().cast::<f32>();
                    } else {
                        extension = 1.0;
                    }

                    for m in moves.sorted().skip(1) {
                        let pv = -self.next(Some(m)).nw(se_depth - 1.0, -se_beta + 1, !cut)?;
                        if pv.score().min(se_beta) >= beta {
                            return Ok(pv.truncate().transpose(m));
                        } else if pv >= se_beta {
                            extension = -2.0 * expected_cut.cast::<f32>();
                            cut = expected_cut;
                            break;
                        } else if pv >= de_beta {
                            extension = extension.min(1.0);
                        } else if pv >= te_beta {
                            extension = extension.min(2.0);
                        }
                    }
                }
            }

            let mut next = self.next(Some(head));
            -next.ab::<IS_PV, _>(depth + extension - 1.0, -beta..-alpha, false)?
        };

        for (index, m) in moves.sorted().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if !IS_PV && !is_check && !tail.is_losing() {
                let scale = Params::lmp_improving(0).mul_add(improving, 1.0);
                if index.cast::<f32>() > Self::lmp(depth) * scale {
                    break;
                }
            }

            let pos = &self.stack.pos;
            let mut lmr = Self::lmr(depth, index);
            let lmr_depth = depth - lmr.clip(0.0, depth.max(1.0) - 1.0);
            let history = self.local.history.get(pos, m);
            let counter = self.stack.reply(1).get(pos, m);
            let is_killer = killer.contains(m);

            if !is_check && !tail.is_losing() && depth < *Params::futility_depth_limit(0) {
                let margin = pos.gain(m) + Self::futility(lmr_depth);
                if self.stack.value(0) + margin.cast::<i16>() <= alpha {
                    continue;
                }
            }

            let mut margin = Self::see_pruning(lmr_depth, m);
            margin = Params::see_margin_is_killer(0).mul_add(is_killer.cast(), margin);
            if !tail.is_losing() && !pos.gaining(m, margin) {
                continue;
            }

            let mut next = self.next(Some(m));
            let gives_check = next.stack.pos.is_check();

            lmr += convolve([
                (1.0, Params::lmr_not_root(..)),
                (IS_PV.cast(), Params::lmr_is_pv(..)),
                (was_pv.cast(), Params::lmr_was_pv(..)),
                (cut.cast(), Params::lmr_is_cut(..)),
                (improving, Params::lmr_improving(..)),
                (is_killer.cast(), Params::lmr_is_killer(..)),
                (is_noisy_node.cast(), Params::lmr_is_noisy_node(..)),
                (gives_check.cast(), Params::lmr_gives_check(..)),
                (history, Params::lmr_history(..)),
                (counter, Params::lmr_counter(..)),
            ]);

            let next_depth = depth.max(1.0) - 1.0;
            let lmr = lmr.clip(*Params::lmr_limit(0), next_depth);
            let mut pv = -next.nw(next_depth - lmr, -alpha, !cut)?.truncate();

            if pv > alpha && lmr > *Params::lmr_threshold(0) {
                pv = -next.nw(next_depth, -alpha, !cut)?.truncate();
            }

            if pv > alpha && pv < beta {
                pv = -next.ab::<IS_PV, _>(next_depth, -beta..-alpha, false)?;
            }

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        let tail = tail.clip(lower, upper);
        let score = ScoreBound::new(bounds, tail.score(), ply);
        let tpos = Transposition::new(score, depth.saturate(), Some(head), was_pv);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, &moves);
            if head.is_quiet() {
                self.stack.killers[ply.cast::<usize>()].insert(head);
            }
        }

        if head.is_quiet() && !score.range(ply).contains(&self.stack.value(0)) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// The root of the principal variation search.
    #[inline(always)]
    fn root(
        &mut self,
        moves: &mut Moves,
        depth: f32,
        bounds: Range<Score>,
    ) -> Result<Pv, Interrupted> {
        let (alpha, beta) = (bounds.start, bounds.end);
        if self.ctrl.check(depth.saturate(), zero(), &self.stack.pv) != Continue {
            return Err(Interrupted);
        }

        let correction = self.correction().cast::<i16>();
        self.stack.values[0] = self.evaluate() + correction;

        moves.sort(|m| {
            if Some(m) == self.stack.pv.head() {
                return Bounded::upper();
            }

            let mut rating = 0.0;
            let pos = &self.stack.pos;
            let history = self.local.history.get(pos, m);
            rating = Params::history_rating(0).mul_add(history, rating);
            if m.is_noisy() && pos.gaining(m, *Params::good_noisy_margin(0)) {
                rating += *Params::good_noisy_rating(0);
                rating += pos.gain(m);
            }

            rating.saturate()
        });

        let mut sorted_moves = moves.sorted();
        let mut head = sorted_moves.next().assume();
        self.stack.nodes = self.ctrl.attention(head);

        let mut next = self.next(Some(head));
        let mut tail = -next.ab::<true, _>(depth - 1.0, -beta..-alpha, false)?;
        drop(next);

        for (index, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let pos = &self.stack.pos;
            let mut lmr = Self::lmr(depth, index);
            let history = self.local.history.get(pos, m);
            self.stack.nodes = self.ctrl.attention(m);

            let mut next = self.next(Some(m));
            let gives_check = next.stack.pos.is_check();

            lmr += convolve([
                (1.0, Params::lmr_is_root(..)),
                (gives_check.cast(), Params::lmr_gives_check(..)),
                (history, Params::lmr_history(..)),
            ]);

            let next_depth = depth.max(1.0) - 1.0;
            let lmr = lmr.clip(*Params::lmr_limit(0), next_depth);
            let mut pv = -next.nw(next_depth - lmr, -alpha, true)?.truncate();

            if pv > alpha && lmr > *Params::lmr_threshold(0) {
                pv = -next.nw(next_depth, -alpha, true)?.truncate();
            }

            if pv > alpha && pv < beta {
                pv = -next.ab::<true, _>(next_depth, -beta..-alpha, false)?;
            }

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        let score = ScoreBound::new(bounds, tail.score(), zero());
        let tpos = Transposition::new(score, depth.saturate(), Some(head), true);
        self.shared.tt.store(self.stack.pos.zobrists().hash, tpos);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, moves);
            if head.is_quiet() {
                self.stack.killers[0].insert(head);
            }
        }

        if head.is_quiet() && !score.range(zero()).contains(&self.stack.value(0)) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// An implementation of aspiration windows with iterative deepening.
    #[inline(always)]
    fn aw(&mut self, mut moves: Moves) -> impl Iterator<Item = Info> {
        gen move {
            for depth in Depth::iter() {
                let mut reduction = 0.0;
                let mut window = if self.stack.pv.head().is_some() {
                    *Params::aw_width(0)
                } else {
                    f32::INFINITY
                };

                let mut lower = self.stack.pv.cast::<f32>() - window;
                let mut upper = self.stack.pv.cast::<f32>() + window;

                loop {
                    let bounds = lower.saturate()..upper.saturate();
                    let aw_depth = depth.cast::<f32>() - Params::aw_fh_reduction(2).min(reduction);
                    let Ok(partial) = self.root(&mut moves, aw_depth, bounds) else {
                        return;
                    };

                    match partial.score() {
                        score if (-lower.saturate::<Score>()..Score::upper()).contains(&-score) => {
                            window *= Params::aw_width(1);
                            upper = Params::aw_fl_lerp(0).lerp(lower, upper);
                            lower = score.cast::<f32>() - window;
                            reduction *= Params::aw_fh_reduction(1);
                        }

                        score if (upper.saturate::<Score>()..Score::upper()).contains(&score) => {
                            self.stack.pv = partial;
                            window *= Params::aw_width(2);
                            upper = score.cast::<f32>() + window;
                            reduction += Params::aw_fh_reduction(0);
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
                vt: ValueTable::new(options.hash),
            },
        }
    }

    /// Resets the hash size.
    pub fn set_hash(&mut self, hash: HashSize) {
        self.shared.tt.resize(hash);
        self.shared.vt.resize(hash);
    }

    /// Resets the thread count.
    pub fn set_threads(&mut self, threads: ThreadCount) {
        self.executor = Executor::new(threads);
        self.local.zeroed_in_place(threads.cast());
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
    fn vt_can_be_resized(s: HashSize, t: HashSize) {
        let mut vt = ValueTable::new(s);
        vt.resize(t);
        assert_eq!(vt.len(), ValueTable::new(t).len());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref] mut e: Engine,
        #[filter(#pos.outcome().is_none() && !#pos.is_check())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_decisive())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_losing() && #s < #b)] s: Score,
        cut: bool,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_hm_limit(0));

        let tpos = Transposition::new(ScoreBound::Upper(s), d, Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw(d.cast(), b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_beta_too_low(
        #[by_ref] mut e: Engine,
        #[filter(#pos.outcome().is_none() && !#pos.is_check())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_decisive())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_winning() && #s >= #b)] s: Score,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_hm_limit(0));

        let tpos = Transposition::new(ScoreBound::Lower(s), d, Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw(d.cast(), b, true), Ok(Pv::empty(s)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn nw_returns_transposition_if_exact(
        #[by_ref] mut e: Engine,
        #[filter(#pos.outcome().is_none() && !#pos.is_check())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_decisive())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_decisive())] s: Score,
    ) {
        prop_assume!(pos.halfmoves() as f32 <= *Params::tt_hm_limit(0));

        let tpos = Transposition::new(ScoreBound::Exact(s), d, Some(m), was_pv);
        e.shared.tt.store(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let stack = Stack::new(pos, Pv::new(s, Line::singular(m)));
        let mut searcher = Searcher::new(ctrl, &e.shared, &mut e.local[0], stack);
        searcher.stack.nodes = searcher.ctrl.attention(m);
        assert_eq!(searcher.nw(d.cast(), b, true), Ok(Pv::empty(s)));
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

        assert_eq!(searcher.ab::<true, 1>(d.cast(), b, cut), Err(Interrupted));
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

        assert_eq!(searcher.ab::<true, 1>(d.cast(), b, cut), Err(Interrupted));
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
            searcher.ab::<true, 1>(d.cast(), b, cut),
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
            searcher.ab::<true, 1>(d.cast(), b, cut),
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
