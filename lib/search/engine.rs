use crate::chess::{Move, Position};
use crate::nnue::{Evaluator, Value};
use crate::syzygy::{Syzygy, Wdl};
use crate::util::{Assume, Bounded, Integer, Memory};
use crate::{params::Params, search::*};
use derive_more::with_trait::{Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{cell::SyncUnsafeCell, ops::Range, pin::Pin, ptr::NonNull, time::Duration};

#[cfg(test)]
use proptest::prelude::*;

/// Indicates the search was interrupted .
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Error)]
#[display("the search was interrupted")]
pub struct Interrupted;

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
    searcher: &'a mut Searcher,
    syzygy: &'a Syzygy,
    tt: &'a Memory<Transposition>,
    ctrl: &'a Control,
    nodes: Option<&'a Nodes>,
    replies: [Option<NonNull<Reply>>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
    evaluator: Evaluator,
    pv: Pv,
}

impl<'a> Stack<'a> {
    fn new(
        searcher: &'a mut Searcher,
        syzygy: &'a Syzygy,
        tt: &'a Memory<Transposition>,
        ctrl: &'a Control,
        evaluator: Evaluator,
    ) -> Self {
        Stack {
            searcher,
            syzygy,
            tt,
            ctrl,
            nodes: None,
            replies: [const { None }; Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            value: [Default::default(); Ply::MAX as usize + 1],
            pv: if evaluator.is_check() {
                Pv::empty(Score::mated(Ply::new(0)))
            } else {
                Pv::empty(Score::new(0))
            },
            evaluator,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        depth: Depth,
        bounds: Range<Score>,
        score: Score,
        best: Move,
        moves: &Moves,
        was_pv: bool,
    ) {
        let pos = &*self.evaluator;
        let ply = self.evaluator.ply();
        let value = self.value[ply.cast::<usize>()];
        let score = ScoreBound::new(bounds, score, ply);
        let tpos = Transposition::new(score, depth, Some(best), was_pv);
        self.tt.set(pos.zobrists().hash, tpos);

        if matches!(score, ScoreBound::Lower(_)) {
            if best.is_quiet() {
                self.killers[ply.cast::<usize>()].insert(best);
            }

            let bonus = self.history_bonus(best, depth).saturate();
            self.searcher.history.update(pos, best, bonus);

            let bonus = self.continuation_bonus(best, depth);
            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            reply.update(pos, best, bonus.saturate());

            for m in moves.iter() {
                if m == best {
                    break;
                } else {
                    let penalty = self.history_penalty(m, depth).saturate();
                    self.searcher.history.update(pos, m, penalty);

                    let penalty = self.continuation_penalty(m, depth);
                    let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
                    reply.update(pos, m, penalty.saturate());
                }
            }
        }

        if best.is_quiet() && !pos.is_check() && !score.range(ply).contains(&value) {
            let zobrists = &pos.zobrists();
            let corrections = &mut self.searcher.corrections;
            let diff = score.bound(ply).cast::<i32>() - value.cast::<i32>();

            let gamma = Params::correction_gradient_gamma();
            let delta = Params::correction_gradient_delta();
            let limit = Params::correction_gradient_limit();
            let grad = (gamma * depth.cast::<i32>() + delta).min(limit) / Params::BASE;

            let bonus = (Params::pawn_correction_bonus() * grad * diff / Params::BASE).saturate();
            corrections.pawn.update(pos, zobrists.pawns, bonus);

            let bonus = (Params::minor_correction_bonus() * grad * diff / Params::BASE).saturate();
            corrections.minor.update(pos, zobrists.minor, bonus);

            let bonus = (Params::major_correction_bonus() * grad * diff / Params::BASE).saturate();
            corrections.major.update(pos, zobrists.major, bonus);

            let bonus = (Params::pieces_correction_bonus() * grad * diff / Params::BASE).saturate();
            corrections.white.update(pos, zobrists.white, bonus);
            corrections.black.update(pos, zobrists.black, bonus);
        }
    }

    fn correction(&mut self) -> i32 {
        let zobrists = &self.evaluator.zobrists();
        let corrections = &mut self.searcher.corrections;
        let pawn = corrections.pawn.get(&self.evaluator, zobrists.pawns);
        let minor = corrections.minor.get(&self.evaluator, zobrists.minor);
        let major = corrections.major.get(&self.evaluator, zobrists.major);
        let white = corrections.white.get(&self.evaluator, zobrists.white);
        let black = corrections.black.get(&self.evaluator, zobrists.black);

        let mut correction = 0;
        correction += pawn as i32 * Params::pawn_correction();
        correction += minor as i32 * Params::minor_correction();
        correction += major as i32 * Params::major_correction();
        correction += white as i32 * Params::pieces_correction();
        correction += black as i32 * Params::pieces_correction();
        correction / Params::BASE / Correction::LIMIT as i32
    }

    fn history_bonus(&self, m: Move, depth: Depth) -> i32 {
        let params = [
            Params::noisy_history_bonus_gamma(),
            Params::noisy_history_bonus_delta(),
            Params::quiet_history_bonus_gamma(),
            Params::quiet_history_bonus_delta(),
        ];

        let offset = 2 * m.is_quiet() as usize;
        let (gamma, delta) = (params[offset], params[offset + 1]);
        (gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    fn continuation_bonus(&self, m: Move, depth: Depth) -> i32 {
        let params = [
            Params::noisy_continuation_bonus_gamma(),
            Params::noisy_continuation_bonus_delta(),
            Params::quiet_continuation_bonus_gamma(),
            Params::quiet_continuation_bonus_delta(),
        ];

        let offset = 2 * m.is_quiet() as usize;
        let (gamma, delta) = (params[offset], params[offset + 1]);
        (gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    fn history_penalty(&self, m: Move, depth: Depth) -> i32 {
        let params = [
            Params::noisy_history_penalty_gamma(),
            Params::noisy_history_penalty_delta(),
            Params::quiet_history_penalty_gamma(),
            Params::quiet_history_penalty_delta(),
        ];

        let offset = 2 * m.is_quiet() as usize;
        let (gamma, delta) = (params[offset], params[offset + 1]);
        -(gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    fn continuation_penalty(&self, m: Move, depth: Depth) -> i32 {
        let params = [
            Params::noisy_continuation_penalty_gamma(),
            Params::noisy_continuation_penalty_delta(),
            Params::quiet_continuation_penalty_gamma(),
            Params::quiet_continuation_penalty_delta(),
        ];

        let offset = 2 * m.is_quiet() as usize;
        let (gamma, delta) = (params[offset], params[offset + 1]);
        -(gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    /// A measure for how much the position is improving.
    fn improving(&self) -> i32 {
        if self.evaluator.is_check() {
            return 0;
        }

        let ply = self.evaluator.ply();
        let idx = ply.cast::<usize>();
        let value = self.value[idx];

        let a = ply >= 2 && !self.evaluator[ply - 2].is_check() && value > self.value[idx - 2];
        let b = ply >= 4 && !self.evaluator[ply - 4].is_check() && value > self.value[idx - 4];

        Params::improving_2() * a as i32 + Params::improving_4() * b as i32
    }

    /// The mate distance pruning.
    fn mdp(&self, bounds: &Range<Score>) -> (Score, Score) {
        let ply = self.evaluator.ply();
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    /// Computes the null move pruning reduction.
    fn nmp(&self, surplus: Score, depth: Depth) -> Option<Depth> {
        let gamma = Params::null_move_reduction_gamma();
        let delta = Params::null_move_reduction_delta();
        match Params::BASE * surplus.cast::<i32>() {
            ..0 => None,
            s if s < gamma - delta => None,
            s if s >= 3 * gamma - delta => Some(depth - 3 - depth / 4),
            s => Some(depth - (s + delta) / gamma - depth / 4),
        }
    }

    /// Computes fail-high pruning reduction.
    fn fhp(&self, surplus: Score, depth: Depth) -> Option<Depth> {
        let gamma = Params::fail_high_reduction_gamma();
        let delta = Params::fail_high_reduction_delta();
        match Params::BASE * surplus.cast::<i32>() {
            ..0 => None,
            s if s >= 3 * gamma - delta => Some(depth - 3),
            s => Some(depth - (s + delta) / gamma),
        }
    }

    /// Computes the fail-low pruning reduction.
    fn flp(&self, deficit: Score, depth: Depth) -> Option<Depth> {
        let gamma = Params::fail_low_reduction_gamma();
        let delta = Params::fail_low_reduction_delta();
        match Params::BASE * deficit.cast::<i32>() {
            ..0 => None,
            s if s >= 3 * gamma - delta => Some(depth - 3),
            s => Some(depth - (s + delta) / gamma),
        }
    }

    /// Computes the singular extension margin.
    fn single(&self, depth: Depth) -> i32 {
        let gamma = Params::single_extension_margin_gamma();
        let delta = Params::single_extension_margin_delta();
        (gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    /// Computes the double extension margin.
    fn double(&self, depth: Depth) -> i32 {
        let gamma = Params::double_extension_margin_gamma();
        let delta = Params::double_extension_margin_delta();
        (gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    /// Computes the triple extension margin.
    fn triple(&self, depth: Depth) -> i32 {
        let gamma = Params::triple_extension_margin_gamma();
        let delta = Params::triple_extension_margin_delta();
        (gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    /// Computes the razoring margin.
    fn razoring(&self, depth: Depth) -> i32 {
        let gamma = Params::razoring_margin_gamma();
        let delta = Params::razoring_margin_delta();

        if depth <= 4 {
            (gamma * depth.cast::<i32>() + delta) / Params::BASE
        } else {
            i32::MAX
        }
    }

    /// Computes the reverse futility margin.
    fn rfp(&self, depth: Depth) -> i32 {
        let gamma = Params::reverse_futility_margin_gamma();
        let delta = Params::reverse_futility_margin_delta();

        if depth <= 6 {
            (gamma * depth.cast::<i32>() + delta) / Params::BASE
        } else {
            i32::MAX
        }
    }

    /// Computes the futility margin.
    fn futility(&self, depth: Depth) -> i32 {
        let gamma = Params::futility_margin_gamma();
        let delta = Params::futility_margin_delta();
        gamma * depth.cast::<i32>() + delta
    }

    /// Computes the SEE pruning threshold.
    fn spt(&self, depth: Depth) -> i32 {
        let gamma = Params::see_pruning_gamma();
        let delta = Params::see_pruning_delta();
        delta - gamma * depth.cast::<i32>()
    }

    /// Computes the late move pruning threshold.
    fn lmp(&self, depth: Depth, idx: usize) -> i32 {
        let gamma = Params::late_move_pruning_gamma();
        let delta = Params::late_move_pruning_delta();
        Params::BASE.pow(2) * idx.cast::<i32>() / (delta + gamma * depth.cast::<i32>().pow(2))
    }

    /// Computes the late move reduction.
    fn lmr(&self, depth: Depth, idx: usize) -> i32 {
        let gamma = Params::late_move_reduction_gamma();
        let delta = Params::late_move_reduction_delta();

        let x = idx.max(1).ilog2() as i32;
        let y = depth.get().max(1).ilog2() as i32;
        gamma * x * y + delta
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
        let ply = self.evaluator.ply();
        self.nodes.assume().increment();
        if self.ctrl.check(&self.pv, ply) == ControlFlow::Abort {
            return Err(Interrupted);
        }

        let (alpha, beta) = match self.evaluator.outcome() {
            None => self.mdp(&bounds),
            Some(o) if o.is_draw() => return Ok(Pv::empty(Score::new(0))),
            Some(_) => return Ok(Pv::empty(Score::mated(ply))),
        };

        if alpha >= beta {
            return Ok(Pv::empty(alpha));
        }

        self.value[ply.cast::<usize>()] = self.evaluator.evaluate() + self.correction();
        let transposition = self.tt.get(self.evaluator.zobrists().hash);
        let transposed = match transposition {
            None => Pv::empty(self.value[ply.cast::<usize>()].saturate()),
            Some(t) => t.transpose(ply),
        };

        if depth > 0 {
            depth += self.evaluator.is_check() as i8;
            depth -= transposition.is_none() as i8;
        }

        let is_pv = alpha + 1 < beta;
        let was_pv = is_pv || transposition.as_ref().is_some_and(Transposition::was_pv);

        if let Some(t) = transposition {
            let (lower, upper) = t.score().range(ply).into_inner();

            if let Some(d) = self.fhp(lower - beta, depth) {
                if !is_pv && t.depth() >= d {
                    return Ok(transposed.truncate());
                }
            }

            if let Some(d) = self.flp(alpha - upper, depth) {
                if !is_pv && t.depth() >= d {
                    return Ok(transposed.truncate());
                }
            }
        }

        let (lower, upper) = if depth <= 0 {
            (transposed.score(), Score::upper())
        } else if !is_pv {
            (Score::lower(), Score::upper())
        } else {
            match self.syzygy.wdl_after_zeroing(&self.evaluator) {
                None => (Score::lower(), Score::upper()),
                Some(wdl) => {
                    let score = match wdl {
                        Wdl::Win => ScoreBound::Lower(wdl.to_score(ply)),
                        Wdl::Loss => ScoreBound::Upper(wdl.to_score(ply)),
                        _ => ScoreBound::Exact(wdl.to_score(ply)),
                    };

                    if score.upper(ply) <= alpha || score.lower(ply) >= beta {
                        let transposition = Transposition::new(score, depth, None, was_pv);
                        self.tt.set(self.evaluator.zobrists().hash, transposition);
                        return Ok(transposition.transpose(ply).truncate());
                    }

                    score.range(ply).into_inner()
                }
            }
        };

        let alpha = alpha.max(lower);
        let transposed = transposed.clamp(lower, upper);
        if alpha >= beta || upper <= alpha || lower >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        } else if !is_pv && !self.evaluator.is_check() && depth > 0 {
            if self.value[ply.cast::<usize>()] + self.razoring(depth) <= alpha {
                let pv = self.nw(Depth::new(0), beta, cut)?;
                if pv <= alpha {
                    return Ok(pv);
                }
            }

            if transposed.score() - self.rfp(depth) >= beta {
                return Ok(transposed.truncate());
            } else if let Some(d) = self.nmp(transposed.score() - beta, depth) {
                if self.evaluator.pieces(self.evaluator.turn()).len() > 1 {
                    if d <= 0 || -self.next(None).nw::<0>(d - 1, -beta + 1, !cut)? >= beta {
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

            let mut rating = 0i32;
            rating += Params::history_rating()
                * self.searcher.history.get(&self.evaluator, m).cast::<i32>();

            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            rating += Params::continuation_rating() * reply.get(&self.evaluator, m).cast::<i32>();

            if killer.contains(m) {
                rating += Params::killer_move_bonus();
            } else if !m.is_quiet() {
                let gain = self.evaluator.gain(m);
                if self.evaluator.winning(m, Value::new(1)) {
                    let gamma = Params::winning_rating_gamma();
                    let delta = Params::winning_rating_delta();
                    rating += gamma * gain.cast::<i32>() + delta;
                }
            }

            (rating / Params::BASE).saturate()
        });

        #[allow(clippy::blocks_in_conditions)]
        let (mut head, mut tail) = match { moves.sorted().next() } {
            None => return Ok(transposed.truncate()),
            Some(m) => {
                let mut extension = 0i8;
                if let Some(t) = transposition {
                    if t.score().lower(ply) >= beta && t.depth() >= depth - 3 && depth >= 6 {
                        extension = 2 + m.is_quiet() as i8;
                        let s_depth = (depth - 1) / 2;
                        let s_beta = beta - self.single(depth);
                        let d_beta = beta - self.double(depth);
                        let t_beta = beta - self.triple(depth);
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

        let improving = self.improving();
        for (idx, m) in moves.sorted().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut lmp = Params::late_move_pruning_baseline();
            lmp += is_pv as i32 * Params::late_move_pruning_is_pv();
            lmp += was_pv as i32 * Params::late_move_pruning_was_pv();
            lmp += self.evaluator.is_check() as i32 * Params::late_move_pruning_check();
            lmp += improving * Params::late_move_pruning_improving() / Params::BASE;
            if self.lmp(depth, idx) > lmp {
                break;
            }

            let mut lmr = self.lmr(depth, idx);
            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let continuation = reply.get(&self.evaluator, m).cast::<i32>();
            let history = self.searcher.history.get(&self.evaluator, m).cast::<i32>();

            let mut futility = self.futility(depth - lmr / Params::BASE);
            futility += is_pv as i32 * Params::futility_margin_is_pv();
            futility += was_pv as i32 * Params::futility_margin_was_pv();
            futility += self.evaluator.gain(m).cast::<i32>() * Params::futility_margin_gain();
            futility += killer.contains(m) as i32 * Params::futility_margin_killer();
            futility += self.evaluator.is_check() as i32 * Params::futility_margin_check();
            futility += history * Params::futility_margin_history() / History::LIMIT as i32;
            futility += continuation * Params::futility_margin_continuation() / Reply::LIMIT as i32;
            futility += improving * Params::futility_margin_improving() / Params::BASE;
            if self.value[ply.cast::<usize>()] + futility.max(0) / Params::BASE <= alpha {
                continue;
            }

            let mut spt = self.spt(depth - lmr / Params::BASE);
            spt -= killer.contains(m) as i32 * Params::see_pruning_killer();
            spt -= history * Params::see_pruning_history() / History::LIMIT as i32;
            spt -= continuation * Params::see_pruning_continuation() / Reply::LIMIT as i32;
            if !self.evaluator.winning(m, (spt / Params::BASE).saturate()) {
                continue;
            }

            let mut next = self.next(Some(m));
            lmr += Params::late_move_reduction_baseline();
            lmr -= is_pv as i32 * Params::late_move_reduction_is_pv();
            lmr -= was_pv as i32 * Params::late_move_reduction_was_pv();
            lmr -= killer.contains(m) as i32 * Params::late_move_reduction_killer();
            lmr -= next.evaluator.is_check() as i32 * Params::late_move_reduction_check();
            lmr -= history * Params::late_move_reduction_history() / History::LIMIT as i32;
            lmr -= continuation * Params::late_move_reduction_continuation() / Reply::LIMIT as i32;
            lmr -= improving * Params::late_move_reduction_improving() / Params::BASE;
            lmr += cut as i32 * Params::late_move_reduction_cut();
            lmr += transposed.head().is_some_and(|m| !m.is_quiet()) as i32
                * Params::late_move_reduction_noisy_pv();

            let pv = match -next.nw(depth - lmr.max(0) / Params::BASE - 1, -alpha, !cut)? {
                pv if pv <= alpha || (pv >= beta && lmr < Params::BASE) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        let tail = tail.clamp(lower, upper);
        self.record(depth, bounds, tail.score(), head, &moves, was_pv);
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
        if self.ctrl.check(&self.pv, self.evaluator.ply()) != ControlFlow::Continue {
            return Err(Interrupted);
        }

        moves.sort(|m| {
            if Some(m) == self.pv.head() {
                Bounded::upper()
            } else {
                let mut rating = Bounded::new(0);
                rating += self.searcher.history.get(&self.evaluator, m);
                rating += self.evaluator.gain(m);
                rating
            }
        });

        let mut sorted_moves = moves.sorted();
        let mut head = sorted_moves.next().assume();
        let mut next = self.next(Some(head));
        next.nodes = Some(next.ctrl.attention().nodes(head));
        let mut tail = -next.ab(depth - 1, -beta..-alpha, false)?;
        drop(next);

        for (idx, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut lmp = Params::late_move_pruning_baseline();
            lmp += self.evaluator.is_check() as i32 * Params::late_move_pruning_check();
            if self.lmp(depth, idx) > lmp + Params::late_move_pruning_root() {
                break;
            }

            let mut lmr = self.lmr(depth, idx);
            let history = self.searcher.history.get(&self.evaluator, m).cast::<i32>();

            let mut next = self.next(Some(m));
            lmr += Params::late_move_reduction_baseline();
            lmr -= Params::late_move_reduction_root();
            lmr -= next.evaluator.is_check() as i32 * Params::late_move_reduction_check();
            lmr -= history * Params::late_move_reduction_history() / History::LIMIT as i32;
            lmr += next.pv.head().is_some_and(|m| !m.is_quiet()) as i32
                * Params::late_move_reduction_noisy_pv();

            next.nodes = Some(next.ctrl.attention().nodes(m));
            let pv = match -next.nw(depth - lmr.max(0) / Params::BASE - 1, -alpha, false)? {
                pv if pv <= alpha || (pv >= beta && lmr < Params::BASE) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        self.record(depth, bounds, tail.score(), head, moves, true);
        Ok(tail.transpose(head))
    }

    /// An implementation of aspiration windows with iterative deepening.
    fn aw(&mut self) -> impl Iterator<Item = Info> {
        gen move {
            let clock = self.ctrl.limits().clock;
            let mut moves = Moves::from_iter(self.evaluator.moves().unpack());
            let mut stop = matches!((moves.len(), &clock), (0, _) | (1, Some(_)));
            let mut depth = Depth::new(0);

            self.value[0] = self.evaluator.evaluate();
            if let Some(t) = self.tt.get(self.evaluator.zobrists().hash) {
                if t.best().is_some_and(|m| moves.iter().any(|n| m == n)) {
                    self.pv = t.transpose(Ply::new(0)).truncate();
                }
            }

            if self.pv.head().is_none() {
                if let Some(m) = moves.iter().next() {
                    self.pv = Pv::new(self.value[0].saturate(), Line::singular(m));
                }
            }

            let aw_start = Params::aspiration_window_start();
            let aw_gamma = Params::aspiration_window_gamma();
            let aw_delta = Params::aspiration_window_delta();

            loop {
                let pv = self.pv.clone().truncate();
                yield Info::new(depth, self.ctrl.time(), self.ctrl.nodes(), pv);
                if stop || depth >= Depth::upper() {
                    return;
                }

                depth += 1;
                let mut reduction = 0;
                let mut window = aw_start / Params::BASE;
                let (mut lower, mut upper) = match depth.get() {
                    ..=4 => (Score::lower(), Score::upper()),
                    _ => (self.pv.score() - window, self.pv.score() + window),
                };

                loop {
                    window = (window * aw_gamma + aw_delta) / Params::BASE;
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
    ctrl: Control,
    channel: Option<UnboundedReceiver<Info>>,
    task: Option<Task<'e>>,
}

impl<'e, 'p> Search<'e, 'p> {
    fn new(engine: &'e mut Engine, position: &'p Position, limits: Limits) -> Self {
        Search {
            engine,
            position,
            ctrl: Control::new(position, limits),
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
        let ctrl: &'static Control = unsafe { &*(&self.ctrl as *const _) };
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
            for info in Stack::new(searcher, syzygy, tt, ctrl, evaluator).aw() {
                if idx == 0 {
                    let depth = info.depth();
                    tx.unbounded_send(info).assume();
                    if depth >= ctrl.limits().max_depth() {
                        break;
                    }
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

#[derive(Debug, Default)]
struct Corrections {
    pawn: Correction,
    minor: Correction,
    major: Correction,
    white: Correction,
    black: Correction,
}

#[derive(Debug, Default)]
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
            searchers: (0..options.threads.get())
                .map(|_| Searcher::default())
                .collect(),
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

        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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

        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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

        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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
        let ctrl = Control::new(&pos, Limits::nodes(0));
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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
        let ctrl = Control::new(&pos, Limits::time(Duration::ZERO));
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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
        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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
        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
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
        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.nodes = Some(ctrl.attention().nodes(m));
        assert_eq!(stack.ab::<1>(d, b, cut), Ok(Pv::empty(Score::mated(ply))));
    }

    #[proptest]
    fn aw_extends_time_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let evaluator = Evaluator::new(pos);
        let ctrl = Control::new(&evaluator, Limits::time(Duration::ZERO));
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, evaluator);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_depth_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let evaluator = Evaluator::new(pos);
        let ctrl = Control::new(&evaluator, Limits::depth(Depth::lower()));
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, &ctrl, evaluator);
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
