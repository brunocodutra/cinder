use crate::chess::{Move, Position};
use crate::nnue::{Evaluator, Value};
use crate::syzygy::{Syzygy, Wdl};
use crate::util::{Assume, Bounded, Integer, Memory};
use crate::{params::Params, search::*};
use bytemuck::{Zeroable, try_zeroed_slice_box};
use derive_more::with_trait::{Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{cell::SyncUnsafeCell, ops::Range, pin::Pin, ptr::NonNull, time::Duration};

#[cfg(test)]
use proptest::prelude::*;

#[inline(always)]
fn convolve<const N: usize>(data: [(i64, &[i64]); N]) -> i64 {
    let mut convolution = 0;

    for i in 0..N {
        for j in i..N {
            let &param = data[i].1.get(j - i).assume();
            convolution += param * data[i].0 * data[j].0;
        }
    }

    convolution
}

#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Error)]
#[display("the search was interrupted")]
struct Interrupted;

#[derive(Debug, Deref, DerefMut)]
struct StackGuard<'e, 'a> {
    stack: &'e mut Stack<'a>,
}

impl<'e, 'a> Drop for StackGuard<'e, 'a> {
    #[inline(always)]
    fn drop(&mut self) {
        self.stack.evaluator.pop();
    }
}

#[derive(Debug)]
struct Stack<'a> {
    index: usize,
    searcher: &'a mut Searcher,
    syzygy: &'a Syzygy,
    tt: &'a Memory<Transposition>,
    ctrl: LocalControl<'a>,
    nodes: Option<NonNull<Nodes>>,
    replies: [Option<NonNull<Reply>>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
    eval: [Value; Ply::MAX as usize + 1],
    evaluator: Evaluator,
    pv: Pv,
}

impl<'a> Stack<'a> {
    fn new(
        index: usize,
        searcher: &'a mut Searcher,
        syzygy: &'a Syzygy,
        tt: &'a Memory<Transposition>,
        ctrl: &'a GlobalControl,
        evaluator: Evaluator,
    ) -> Self {
        Stack {
            index,
            searcher,
            syzygy,
            tt,
            ctrl: LocalControl::new(ctrl),
            nodes: None,
            replies: [const { None }; Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            value: [Default::default(); Ply::MAX as usize + 1],
            eval: [Default::default(); Ply::MAX as usize + 1],
            pv: if evaluator.is_check() {
                Pv::empty(Score::mated(Ply::new(0)))
            } else {
                Pv::empty(Score::new(0))
            },
            evaluator,
        }
    }

    #[inline(always)]
    fn transposition(&self) -> Option<Transposition> {
        let tpos = self.tt.get(self.evaluator.zobrists().hash)?;
        if tpos.best().is_none_or(|m| self.evaluator.is_legal(m)) {
            Some(tpos)
        } else {
            None
        }
    }

    /// The mate distance pruning.
    #[inline(always)]
    fn mdp(&self, bounds: &Range<Score>) -> (Score, Score) {
        let ply = self.evaluator.ply();
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    #[inline(always)]
    fn correction(&mut self) -> i64 {
        let zobrists = &self.evaluator.zobrists();
        let corrections = &mut self.searcher.corrections;
        let pawns = corrections.pawns.get(&self.evaluator, zobrists.pawns);
        let minor = corrections.minor.get(&self.evaluator, zobrists.minor);
        let major = corrections.major.get(&self.evaluator, zobrists.major);
        let white = corrections.white.get(&self.evaluator, zobrists.white);
        let black = corrections.black.get(&self.evaluator, zobrists.black);

        let mut correction = 0;
        correction += pawns as i64 * Params::pawns_correction()[0];
        correction += minor as i64 * Params::minor_correction()[0];
        correction += major as i64 * Params::major_correction()[0];
        correction += white as i64 * Params::pieces_correction()[0];
        correction += black as i64 * Params::pieces_correction()[0];
        correction / Correction::LIMIT as i64 / Params::BASE
    }

    #[inline(always)]
    fn update_correction(&mut self, depth: Depth, score: ScoreBound) {
        let pos = &*self.evaluator;
        let ply = self.evaluator.ply();
        let zobrists = &pos.zobrists();
        let log_depth = depth.get().max(1).ilog2();
        let diff = score.bound(ply).cast::<i64>() - self.value[ply.cast::<usize>()].cast::<i64>();

        let grad = convolve([
            (log_depth as i64, &Params::correction_gradient_depth()),
            (1, &Params::correction_gradient_scalar()),
        ]) / Params::BASE;

        let corrections = &mut self.searcher.corrections;
        let bonus = (Params::pawns_correction_bonus()[0] * diff * grad / Params::BASE).saturate();
        corrections.pawns.update(pos, zobrists.pawns, bonus);

        let bonus = (Params::minor_correction_bonus()[0] * diff * grad / Params::BASE).saturate();
        corrections.minor.update(pos, zobrists.minor, bonus);

        let bonus = (Params::major_correction_bonus()[0] * diff * grad / Params::BASE).saturate();
        corrections.major.update(pos, zobrists.major, bonus);

        let bonus = (Params::pieces_correction_bonus()[0] * diff * grad / Params::BASE).saturate();
        corrections.white.update(pos, zobrists.white, bonus);
        corrections.black.update(pos, zobrists.black, bonus);
    }

    #[inline(always)]
    fn history_bonus(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::history_bonus_depth()),
            (1, &Params::history_bonus_scalar()),
        ])
    }

    #[inline(always)]
    fn continuation_bonus(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::continuation_bonus_depth()),
            (1, &Params::continuation_bonus_scalar()),
        ])
    }

    #[inline(always)]
    fn history_penalty(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::history_penalty_depth()),
            (1, &Params::history_penalty_scalar()),
        ])
    }

    #[inline(always)]
    fn continuation_penalty(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::continuation_penalty_depth()),
            (1, &Params::continuation_penalty_scalar()),
        ])
    }

    #[inline(always)]
    fn update_history(&mut self, depth: Depth, best: Move, moves: &Moves) {
        let pos = &*self.evaluator;
        let ply = self.evaluator.ply();

        let bonus = Self::history_bonus(depth) / Params::BASE;
        self.searcher.history.update(pos, best, bonus.saturate());

        let bonus = Self::continuation_bonus(depth) / Params::BASE;
        let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
        reply.update(pos, best, bonus.saturate());

        for m in moves.iter() {
            if m == best {
                break;
            } else {
                let penalty = Self::history_penalty(depth) / Params::BASE;
                self.searcher.history.update(pos, m, penalty.saturate());

                let penalty = Self::continuation_penalty(depth) / Params::BASE;
                let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
                reply.update(pos, m, penalty.saturate());
            }
        }
    }

    /// A measure for how much the position is improving.
    #[inline(always)]
    fn improving(&self) -> i64 {
        if self.evaluator.is_check() {
            return 0;
        }

        let ply = self.evaluator.ply();
        let idx = ply.cast::<usize>();
        let value = self.value[idx];

        let a = ply >= 2 && !self.evaluator[ply - 2].is_check() && value > self.value[idx - 2];
        let b = ply >= 4 && !self.evaluator[ply - 4].is_check() && value > self.value[idx - 4];

        convolve([
            (a as _, &Params::improving_2()),
            (b as _, &Params::improving_4()),
        ])
    }

    /// Computes the null move reduction.
    #[inline(always)]
    fn nmr(depth: Depth, surplus: Score) -> Option<i64> {
        let gamma = Params::null_move_reduction_gamma()[0];
        let delta = Params::null_move_reduction_delta()[0];

        match depth.cast::<i64>() {
            ..3 => None,
            d @ 3.. => match Params::BASE * surplus.cast::<i64>() {
                s if s < gamma - delta => None,
                s if s >= 3 * gamma - delta => Some(3 + d / 4),
                s => Some((s + delta) / gamma + d / 4),
            },
        }
    }

    /// Computes the null move pruning margin.
    #[inline(always)]
    fn nmp(depth: Depth) -> Option<i64> {
        match depth.cast() {
            ..1 | 5.. => None,
            d @ 1..5 => Some(convolve([
                (d, &Params::null_move_pruning_depth()),
                (1, &Params::null_move_pruning_scalar()),
            ])),
        }
    }

    /// Computes the fail-low pruning reduction.
    #[inline(always)]
    fn flp(depth: Depth) -> Option<i64> {
        match depth.cast() {
            5.. => None,
            ..1 => Some(0),
            d @ 1..5 => Some(convolve([
                (d, &Params::fail_low_pruning_depth()),
                (1, &Params::fail_low_pruning_scalar()),
            ])),
        }
    }

    /// Computes fail-high pruning reduction.
    #[inline(always)]
    fn fhp(depth: Depth) -> Option<i64> {
        match depth.cast() {
            7.. => None,
            ..1 => Some(0),
            d @ 1..7 => Some(convolve([
                (d, &Params::fail_high_pruning_depth()),
                (1, &Params::fail_high_pruning_scalar()),
            ])),
        }
    }

    /// Computes the razoring margin.
    #[inline(always)]
    fn razoring(depth: Depth) -> Option<i64> {
        match depth.cast() {
            ..1 | 5.. => None,
            d @ 1..5 => Some(convolve([
                (d, &Params::razoring_margin_depth()),
                (1, &Params::razoring_margin_scalar()),
            ])),
        }
    }

    /// Computes the reverse futility margin.
    #[inline(always)]
    fn rfp(depth: Depth) -> Option<i64> {
        match depth.cast() {
            ..1 | 9.. => None,
            d @ 1..9 => Some(convolve([
                (d, &Params::reverse_futility_margin_depth()),
                (1, &Params::reverse_futility_margin_scalar()),
            ])),
        }
    }

    /// Computes the futility margin.
    #[inline(always)]
    fn futility(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::futility_margin_depth()),
            (1, &Params::futility_margin_scalar()),
        ])
    }

    /// Computes the probcut margin.
    #[inline(always)]
    fn probcut(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::probcut_margin_depth()),
            (1, &Params::probcut_margin_scalar()),
        ])
    }

    /// Computes the singular extension margin.
    #[inline(always)]
    fn single(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::single_extension_margin_depth()),
            (1, &Params::single_extension_margin_scalar()),
        ])
    }

    /// Computes the double extension margin.
    #[inline(always)]
    fn double(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::double_extension_margin_depth()),
            (1, &Params::double_extension_margin_scalar()),
        ])
    }

    /// Computes the triple extension margin.
    #[inline(always)]
    fn triple(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::triple_extension_margin_depth()),
            (1, &Params::triple_extension_margin_scalar()),
        ])
    }

    /// Computes the noisy SEE pruning margin.
    #[inline(always)]
    fn nsp(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::noisy_see_pruning_depth()),
            (1, &Params::noisy_see_pruning_scalar()),
        ])
    }

    /// Computes the quiet SEE pruning margin.
    #[inline(always)]
    fn qsp(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::quiet_see_pruning_depth()),
            (1, &Params::quiet_see_pruning_scalar()),
        ])
    }

    /// Computes the late move pruning threshold.
    #[inline(always)]
    fn lmp(depth: Depth) -> i64 {
        convolve([
            (depth.cast(), &Params::late_move_pruning_depth()),
            (1, &Params::late_move_pruning_scalar()),
        ])
    }

    /// Computes the late move reduction.
    #[inline(always)]
    fn lmr(depth: Depth, index: usize) -> i64 {
        let log_depth = depth.get().max(1).ilog2();
        let log_index = index.max(1).ilog2();

        convolve([
            (log_depth.cast(), &Params::late_move_reduction_depth()),
            (log_index.cast(), &Params::late_move_reduction_index()),
            (1, &Params::late_move_reduction_scalar()),
        ])
    }

    #[must_use]
    fn next(&mut self, m: Option<Move>) -> StackGuard<'_, 'a> {
        self.replies[self.evaluator.ply().cast::<usize>()] = m.map(|m| {
            let reply = self.searcher.continuation.reply(&self.evaluator, m);
            NonNull::from_mut(reply)
        });

        self.evaluator.push(m);
        self.tt.prefetch(self.evaluator.zobrists().hash);

        StackGuard { stack: self }
    }

    /// The zero-window alpha-beta search.
    fn nw<const N: usize>(
        &mut self,
        depth: Depth,
        beta: Score,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        self.ab(depth, beta - 1..beta, cut)
    }

    /// The alpha-beta search.
    fn ab<const N: usize>(
        &mut self,
        depth: Depth,
        bounds: Range<Score>,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        if self.evaluator.ply().cast::<usize>() < N && depth > 0 && bounds.start + 1 < bounds.end {
            self.pvs(depth, bounds, cut)
        } else {
            Ok(self.pvs::<0>(depth, bounds, cut)?.truncate())
        }
    }

    /// The principal variation search.
    fn pvs<const N: usize>(
        &mut self,
        mut depth: Depth,
        bounds: Range<Score>,
        mut cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        self.nodes.update(1);
        if self.ctrl.check(&self.pv, &self.evaluator) == ControlFlow::Abort {
            return Err(Interrupted);
        }

        let ply = self.evaluator.ply();
        let (alpha, beta) = match self.evaluator.outcome() {
            None => self.mdp(&bounds),
            Some(o) if o.is_draw() => return Ok(Pv::empty(Score::new(0))),
            Some(_) => return Ok(Pv::empty(Score::mated(ply))),
        };

        if alpha >= beta {
            return Ok(Pv::empty(alpha));
        }

        let correction = self.correction();
        self.eval[ply.cast::<usize>()] = self.evaluator.evaluate();
        self.value[ply.cast::<usize>()] = self.eval[ply.cast::<usize>()] + correction;

        let is_check = self.evaluator.is_check();
        let transposition = self.transposition();
        let transposed = match transposition {
            None => Pv::empty(self.value[ply.cast::<usize>()].saturate()),
            Some(t) => t.transpose(ply),
        };

        if depth > 0 {
            depth += is_check as i8;
            depth -= transposition.is_none() as i8;
        }

        let is_pv = alpha + 1 < beta;
        let was_pv = is_pv || transposition.as_ref().is_some_and(Transposition::was_pv);
        let is_noisy_pv = transposition.is_some_and(|t| {
            t.best().is_some_and(|m| !m.is_quiet()) && !matches!(t.score(), ScoreBound::Upper(_))
        });

        if !is_pv && self.evaluator.halfmoves() < 90 {
            if let Some(t) = transposition {
                let (lower, upper) = t.score().range(ply).into_inner();

                if let Some(margin) = Self::flp(depth - t.depth()) {
                    if upper + margin / Params::BASE <= alpha {
                        return Ok(transposed.truncate());
                    }
                }

                if let Some(margin) = Self::fhp(depth - t.depth()) {
                    if lower - margin / Params::BASE >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let (lower, upper) = if depth <= 0 {
            (transposed.score(), Score::upper())
        } else {
            match self.syzygy.wdl_after_zeroing(&self.evaluator) {
                None => (Score::lower(), Score::upper()),
                Some(wdl) => {
                    let score = match wdl {
                        Wdl::Win => ScoreBound::Lower(Score::upper()),
                        Wdl::Loss => ScoreBound::Upper(Score::lower()),
                        _ => ScoreBound::Exact(Score::new(0)),
                    };

                    if score.upper(ply) <= alpha || score.lower(ply) >= beta {
                        let transposition = Transposition::new(score, depth + 4, None, was_pv);
                        self.tt.set(self.evaluator.zobrists().hash, transposition);
                        return Ok(transposition.transpose(ply).truncate());
                    }

                    score.range(ply).into_inner()
                }
            }
        };

        let alpha = alpha.max(lower);
        let improving = self.improving();
        let transposed = transposed.clamp(lower, upper);
        if alpha >= beta || upper <= alpha || lower >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        } else if !is_pv && !is_check && depth > 0 {
            if let Some(margin) = Self::razoring(depth) {
                if self.value[ply.cast::<usize>()] + margin / Params::BASE <= alpha {
                    let pv = self.nw(Depth::new(0), beta, cut)?;
                    if pv <= alpha {
                        return Ok(pv);
                    }
                }
            }

            if let Some(mut margin) = Self::rfp(depth) {
                margin += improving * Params::reverse_futility_margin_improving()[0] / Params::BASE;
                if transposed.score() - margin / Params::BASE >= beta {
                    return Ok(transposed.truncate());
                }
            }

            if self.evaluator.pieces(self.evaluator.turn()).len() > 1 {
                if let Some(margin) = Self::nmp(depth) {
                    if transposed.score() - margin / Params::BASE >= beta {
                        return Ok(transposed.truncate());
                    }
                }

                if let Some(r) = Self::nmr(depth, transposed.score() - beta) {
                    if -self.next(None).nw::<0>(depth - r - 1, -beta + 1, !cut)? >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let move_pack = self.evaluator.moves();
        let mut moves = Moves::from_iter(move_pack.unpack_if(|ms| depth > 0 || !ms.is_quiet()));
        let killer = self.killers[ply.cast::<usize>()];

        moves.sort(|m| {
            if Some(m) == transposed.head() {
                return Bounded::upper();
            }

            let mut rating = killer.contains(m) as i64 * Params::killer_move_bonus()[0];
            let history = self.searcher.history.get(&self.evaluator, m).cast::<i64>();
            rating += history * Params::history_rating()[0] / History::LIMIT as i64;

            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let counter = reply.get(&self.evaluator, m).cast::<i64>();
            rating += counter * Params::counter_rating()[0] / History::LIMIT as i64;

            if !m.is_quiet() {
                let gain = self.evaluator.gain(m);
                let margin = Params::winning_rating_margin()[0] / Params::BASE;
                if self.evaluator.winning(m, margin.saturate()) {
                    rating += convolve([
                        (gain.cast(), &Params::winning_rating_gain()),
                        (1, &Params::winning_rating_scalar()),
                    ]);
                }
            }

            (rating / Params::BASE).saturate()
        });

        if let Some(t) = transposition {
            let p_beta = beta + Self::probcut(depth) / Params::BASE;
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

                    let margin = p_beta - self.value[ply.cast::<usize>()];
                    if !self.evaluator.winning(m, margin.saturate()) {
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
                        let transposition = Transposition::new(score, p_depth, Some(m), was_pv);
                        self.tt.set(self.evaluator.zobrists().hash, transposition);
                        return Ok(pv.transpose(m));
                    }
                }
            }
        }

        #[allow(clippy::blocks_in_conditions)]
        let (mut head, mut tail) = match { moves.sorted().next() } {
            None => return Ok(transposed.truncate()),
            Some(m) => {
                let mut extension = 0i8;
                if let Some(t) = transposition {
                    if t.score().lower(ply) >= beta && t.depth() >= depth - 3 && depth >= 6 {
                        extension = 2 + m.is_quiet() as i8;
                        let s_depth = (depth - 1) / 2;
                        let s_beta = beta - Self::single(depth) / Params::BASE;
                        let d_beta = beta - Self::double(depth) / Params::BASE;
                        let t_beta = beta - Self::triple(depth) / Params::BASE;
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

            let mut lmp = Params::late_move_pruning_baseline()[0];
            lmp += is_pv as i64 * Params::late_move_pruning_is_pv()[0];
            lmp += was_pv as i64 * Params::late_move_pruning_was_pv()[0];
            lmp += is_check as i64 * Params::late_move_pruning_is_check()[0];
            lmp += improving * Params::late_move_pruning_improving()[0] / Params::BASE;
            if index.cast::<i64>() * Params::BASE.pow(2) > Self::lmp(depth) * lmp {
                break;
            }

            let mut lmr = Self::lmr(depth, index);
            let lmr_depth = depth - lmr / Params::BASE;
            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let counter = reply.get(&self.evaluator, m).cast::<i64>();
            let history = self.searcher.history.get(&self.evaluator, m).cast::<i64>();
            let gain = self.evaluator.gain(m).cast::<i64>();
            let is_killer = killer.contains(m);
            let is_quiet = m.is_quiet();

            let mut futility = Self::futility(lmr_depth);
            futility += is_pv as i64 * Params::futility_margin_is_pv()[0];
            futility += was_pv as i64 * Params::futility_margin_was_pv()[0];
            futility += is_check as i64 * Params::futility_margin_is_check()[0];
            futility += is_killer as i64 * Params::futility_margin_is_killer()[0];

            futility += improving * Params::futility_margin_improving()[0] / Params::BASE;
            futility += gain * Params::futility_margin_gain()[0];

            if self.value[ply.cast::<usize>()] + futility.max(0) / Params::BASE <= alpha {
                continue;
            }

            let mut spt = if is_quiet {
                Self::qsp(lmr_depth)
            } else {
                Self::nsp(depth)
            };

            spt += is_killer as i64 * Params::see_pruning_is_killer()[0];
            if !self.evaluator.winning(m, (spt / Params::BASE).saturate()) {
                continue;
            }

            let mut next = self.next(Some(m));
            let gives_check = next.evaluator.is_check();

            lmr += Params::late_move_reduction_baseline()[0];
            lmr += is_pv as i64 * Params::late_move_reduction_is_pv()[0];
            lmr += was_pv as i64 * Params::late_move_reduction_was_pv()[0];
            lmr += gives_check as i64 * Params::late_move_reduction_gives_check()[0];
            lmr += is_noisy_pv as i64 * Params::late_move_reduction_is_noisy_pv()[0];
            lmr += is_killer as i64 * Params::late_move_reduction_is_killer()[0];
            lmr += cut as i64 * Params::late_move_reduction_cut()[0];

            lmr += improving * Params::late_move_reduction_improving()[0] / Params::BASE;
            lmr += history * Params::late_move_reduction_history()[0] / History::LIMIT as i64;
            lmr += counter * Params::late_move_reduction_counter()[0] / History::LIMIT as i64;

            let pv = match -next.nw(depth - lmr.max(0) / Params::BASE - 1, -alpha, !cut)? {
                pv if pv <= alpha || (pv >= beta && lmr < Params::BASE) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
                is_noisy_node |= !head.is_quiet() && tail > alpha;
            }
        }

        let tail = tail.clamp(lower, upper);
        let score = ScoreBound::new(bounds, tail.score(), ply);
        let transposition = Transposition::new(score, depth, Some(head), was_pv);
        self.tt.set(self.evaluator.zobrists().hash, transposition);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, &moves);
            if head.is_quiet() {
                self.killers[ply.cast::<usize>()].insert(head);
            }
        }

        let value = self.value[ply.cast::<usize>()];
        if !is_noisy_node && !score.range(ply).contains(&value) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// The root of the principal variation search.
    fn root(
        &mut self,
        moves: &mut Moves,
        depth: Depth,
        bounds: Range<Score>,
    ) -> Result<Pv, Interrupted> {
        let (alpha, beta) = (bounds.start, bounds.end);
        match self.ctrl.check(&self.pv, &self.evaluator) {
            ControlFlow::Stop if self.index == 0 => return Err(Interrupted),
            ControlFlow::Abort => return Err(Interrupted),
            _ => {}
        }

        let is_check = self.evaluator.is_check();
        let is_noisy_pv = self.pv.head().is_some_and(|m| !m.is_quiet());
        self.value[0] = self.eval[0] + self.correction();

        moves.sort(|m| {
            if Some(m) == self.pv.head() {
                return Bounded::upper();
            }

            let mut rating = 0i64;
            let history = self.searcher.history.get(&self.evaluator, m).cast::<i64>();
            rating += history * Params::history_rating()[0] / History::LIMIT as i64;

            if !m.is_quiet() {
                let gain = self.evaluator.gain(m);
                let margin = Params::winning_rating_margin()[0] / Params::BASE;
                if self.evaluator.winning(m, margin.saturate()) {
                    rating += convolve([
                        (gain.cast(), &Params::winning_rating_gain()),
                        (1, &Params::winning_rating_scalar()),
                    ]);
                }
            }

            (rating / Params::BASE).saturate()
        });

        let mut sorted_moves = moves.sorted();
        let mut head = sorted_moves.next().assume();
        self.nodes = Some(NonNull::from_mut(
            self.ctrl.attention().nodes(&self.evaluator, head),
        ));

        let mut tail = -self.next(Some(head)).ab(depth - 1, -beta..-alpha, false)?;

        let mut is_noisy_node = is_check || is_noisy_pv;
        is_noisy_node |= !head.is_quiet() && tail > alpha;
        for (index, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut lmp = Params::late_move_pruning_is_root()[0];
            lmp += is_check as i64 * Params::late_move_pruning_is_check()[0];
            if index.cast::<i64>() * Params::BASE.pow(2) > Self::lmp(depth) * lmp {
                break;
            }

            let history = self.searcher.history.get(&self.evaluator, m).cast::<i64>();

            self.nodes = Some(NonNull::from_mut(
                self.ctrl.attention().nodes(&self.evaluator, m),
            ));

            let mut next = self.next(Some(m));
            let gives_check = next.evaluator.is_check();

            let mut lmr = Self::lmr(depth, index);
            lmr += Params::late_move_reduction_is_root()[0];
            lmr += gives_check as i64 * Params::late_move_reduction_gives_check()[0];
            lmr += is_noisy_pv as i64 * Params::late_move_reduction_is_noisy_pv()[0];
            lmr += history * Params::late_move_reduction_history()[0] / History::LIMIT as i64;

            let pv = match -next.nw(depth - lmr.max(0) / Params::BASE - 1, -alpha, false)? {
                pv if pv <= alpha || (pv >= beta && lmr < Params::BASE) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
                is_noisy_node |= !head.is_quiet() && tail > alpha;
            }
        }

        let score = ScoreBound::new(bounds, tail.score(), Ply::new(0));
        let transposition = Transposition::new(score, depth, Some(head), true);
        self.tt.set(self.evaluator.zobrists().hash, transposition);

        if matches!(score, ScoreBound::Lower(_)) {
            self.update_history(depth, head, moves);
            if head.is_quiet() {
                self.killers[0].insert(head);
            }
        }

        let value = self.value[0];
        if !is_noisy_node && !score.range(Ply::new(0)).contains(&value) {
            self.update_correction(depth, score);
        }

        Ok(tail.transpose(head))
    }

    /// An implementation of aspiration windows with iterative deepening.
    fn aw(&mut self) -> impl Iterator<Item = Info> {
        gen move {
            let clock = self.ctrl.limits().clock;
            let mut moves = Moves::from_iter(self.evaluator.moves().unpack());
            let mut stop = matches!((moves.len(), &clock), (0, _) | (1, Some(_)));
            let mut depth = Depth::new(0);

            self.eval[0] = self.evaluator.evaluate();
            self.value[0] = self.eval[0] + self.correction();
            if let Some(t) = self.transposition().filter(|t| t.best().is_some()) {
                self.pv = t.transpose(Ply::new(0)).truncate();
            } else if let Some(m) = moves.iter().next() {
                self.pv = Pv::new(self.value[0].saturate(), Line::singular(m));
            }

            loop {
                if self.index == 0 {
                    let pv = self.pv.clone().truncate();
                    yield Info::new(depth, self.ctrl.elapsed(), self.ctrl.visited(), pv);
                }

                stop |= self.index == 0 && depth >= self.ctrl.limits().max_depth();
                if stop || depth >= Depth::upper() {
                    return;
                }

                depth += 1;
                let mut reduction = 0;
                let mut window = Params::aspiration_window_baseline()[0] / Params::BASE;
                let (mut lower, mut upper) = match depth.get() {
                    ..=4 => (Score::lower(), Score::upper()),
                    _ => (self.pv.score() - window, self.pv.score() + window),
                };

                loop {
                    window = convolve([
                        (window, &Params::aspiration_window_exponent()),
                        (1, &Params::aspiration_window_scalar()),
                    ]) / Params::BASE;

                    let partial = match self.root(&mut moves, depth - reduction, lower..upper) {
                        Err(_) => break stop = true,
                        Ok(pv) => pv,
                    };

                    match partial.score() {
                        score if (-lower..Score::upper()).contains(&-score) => {
                            upper = lower / 2 + upper / 2;
                            lower = score - window;
                            reduction = 0;
                        }

                        score if (upper..Score::upper()).contains(&score) => {
                            upper = score + window;
                            self.pv = partial;
                            reduction += 1;
                        }

                        _ => break self.pv = partial,
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
    position: &'p Position,
    ctrl: GlobalControl,
    channel: Option<UnboundedReceiver<Info>>,
    task: Option<Task<'e>>,
}

impl<'e, 'p> Search<'e, 'p> {
    fn new(engine: &'e mut Engine, position: &'p Position, limits: Limits) -> Self {
        Search {
            engine,
            position,
            ctrl: GlobalControl::new(position, limits),
            channel: None,
            task: None,
        }
    }

    /// Aborts the search.
    ///
    /// Returns true if the search had not already been aborted.
    #[inline(always)]
    pub fn abort(&self) {
        self.ctrl.abort();
    }
}

impl<'e, 'p> Drop for Search<'e, 'p> {
    fn drop(&mut self) {
        if let Some(t) = self.task.take() {
            self.abort();
            drop(t);
        }
    }
}

impl<'e, 'p> FusedStream for Search<'e, 'p> {
    fn is_terminated(&self) -> bool {
        self.channel
            .as_ref()
            .is_some_and(FusedStream::is_terminated)
    }
}

impl<'e, 'p> Stream for Search<'e, 'p> {
    type Item = Info;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(rx) = &mut self.channel {
            return rx.poll_next_unpin(cx);
        }

        let executor: &mut Executor = unsafe { &mut *(&mut self.engine.executor as *mut _) };
        let syzygy: &'static Syzygy = unsafe { &*(&self.engine.syzygy as *const _) };
        let tt: &'static Memory<Transposition> = unsafe { &*(&self.engine.tt as *const _) };
        let ctrl: &'static GlobalControl = unsafe { &*(&self.ctrl as *const _) };
        let position: &'static Position = unsafe { &*(self.position as *const _) };
        let searchers: &'static [SyncUnsafeCell<Searcher>] =
            unsafe { &*(&*self.engine.searchers as *const _ as *const [SyncUnsafeCell<Searcher>]) };

        let (tx, rx) = unbounded();
        self.channel = Some(rx);
        if let Some(pv) = syzygy.best(position) {
            let info = Info::new(Depth::new(0), Duration::ZERO, 0, pv.truncate());
            return Poll::Ready(Some(info));
        }

        self.task = Some(executor.execute(move |idx| {
            let evaluator = Evaluator::new(position.clone());
            let searcher = unsafe { &mut *searchers.get(idx).assume().get() };
            for info in Stack::new(idx, searcher, syzygy, tt, ctrl, evaluator).aw() {
                if idx == 0 {
                    tx.unbounded_send(info).assume();
                }
            }

            if idx == 0 {
                tx.close_channel();
                ctrl.abort();
            }
        }));

        self.poll_next(cx)
    }
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
struct Searcher {
    history: History,
    continuation: Continuation,
    corrections: Corrections,
}

/// A chess engine.
#[derive(Debug)]
pub struct Engine {
    tt: Memory<Transposition>,
    syzygy: Syzygy,
    executor: Executor,
    searchers: Box<[Searcher]>,
}

#[cfg(test)]
impl Arbitrary for Engine {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
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
            tt: Memory::new(options.hash.get()),
            syzygy: Syzygy::new(&options.syzygy),
            executor: Executor::new(options.threads),
            searchers: try_zeroed_slice_box(options.threads.cast()).assume(),
        }
    }

    /// Initiates a [`Search`].
    pub fn search<'e, 'p>(&'e mut self, pos: &'p Position, limits: Limits) -> Search<'e, 'p> {
        Search::new(self, pos, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on_stream;
    use proptest::{prop_assume, sample::Selector};
    use std::{thread, time::Duration};
    use test_strategy::proptest;

    #[proptest]
    fn hash_is_an_upper_limit_for_table_size(o: Options) {
        let e = Engine::with_options(&o);
        prop_assume!(e.tt.capacity() > 1);
        assert!(e.tt.size() <= o.hash.get());
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_low(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(#b.mate() == Mate::None)] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(#s.mate() == Mate::None && #s >= #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), Depth::upper(), Some(m), was_pv);
        e.tt.set(pos.zobrists().hash, tpos);

        let ctrl = GlobalControl::new(&pos, Limits::none());
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(#b.mate() == Mate::None)] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(#s.mate() == Mate::None && #s < #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), Depth::upper(), Some(m), was_pv);
        e.tt.set(pos.zobrists().hash, tpos);

        let ctrl = GlobalControl::new(&pos, Limits::none());
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(#b.mate() == Mate::None)] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(#s.mate() == Mate::None)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), Depth::upper(), Some(m), was_pv);
        e.tt.set(pos.zobrists().hash, tpos);

        let ctrl = GlobalControl::new(&pos, Limits::none());
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn ab_aborts_if_maximum_number_of_nodes_visited(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = GlobalControl::new(&pos, Limits::nodes(0));
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_aborts_if_time_is_up(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        stack.pv = stack.pv.transpose(m);
        thread::sleep(Duration::from_millis(1));
        assert_eq!(stack.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_can_be_aborted_upon_request(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = GlobalControl::new(&pos, Limits::none());
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        stack.pv = stack.pv.transpose(m);
        ctrl.abort();
        assert_eq!(stack.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        mut e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = GlobalControl::new(&pos, Limits::none());
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        assert_eq!(stack.ab::<1>(d, b, cut), Ok(Pv::empty(Score::new(0))));
    }

    #[proptest]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        mut e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ply = pos.ply();
        let ctrl = GlobalControl::new(&pos, Limits::none());
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(NonNull::from_mut(
            stack.ctrl.attention().nodes(&stack.evaluator, m),
        ));

        assert_eq!(stack.ab::<1>(d, b, cut), Ok(Pv::empty(Score::mated(ply))));
    }

    #[proptest]
    fn aw_extends_time_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let evaluator = Evaluator::new(pos);
        let ctrl = GlobalControl::new(&evaluator, Limits::time(Duration::ZERO));
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, evaluator);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_depth_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let evaluator = Evaluator::new(pos);
        let ctrl = GlobalControl::new(&evaluator, Limits::depth(Depth::lower()));
        let mut stack = Stack::new(0, &mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, evaluator);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn search_returns_pvs_that_improve_monotonically(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        d: Depth,
    ) {
        let infos = block_on_stream(e.search(&pos, Limits::depth(d)));
        assert!(infos.map(|i| (i.depth(), i.score())).is_sorted());
    }
}
