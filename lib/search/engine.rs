use crate::chess::{Move, Position};
use crate::nnue::{Evaluator, Value};
use crate::search::{ControlFlow::*, *};
use crate::util::{Assume, Bits, Bounded, Float, Int, Memory, Slice, Vault};
use crate::{params::Params, syzygy::Syzygy};
use bytemuck::{Zeroable, fill_zeroes, zeroed};
use derive_more::with_trait::{Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::cell::SyncUnsafeCell;
use std::task::{Context, Poll};
use std::{mem::swap, ops::Range, path::Path, pin::Pin, ptr::NonNull, slice, time::Duration};

#[cfg(test)]
use proptest::prelude::*;

#[inline(always)]
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
        self.stack.pos.pop();
    }
}

#[derive(Debug)]
struct Stack<'a> {
    searcher: &'a mut Searcher,
    syzygy: &'a Syzygy,
    tt: &'a Memory<Transposition>,
    ctrl: LocalControl<'a>,
    pos: Evaluator,
    nodes: Option<NonNull<Nodes>>,
    replies: [Option<NonNull<Reply>>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
    eval: [Value; Ply::MAX as usize + 1],
    pv: Pv,
}

impl<'a> Stack<'a> {
    fn new(
        searcher: &'a mut Searcher,
        syzygy: &'a Syzygy,
        tt: &'a Memory<Transposition>,
        ctrl: LocalControl<'a>,
        pos: Evaluator,
    ) -> Self {
        let pv = if pos.is_check() {
            Pv::empty(Score::mated(Ply::new(0)))
        } else {
            Pv::empty(Score::new(0))
        };

        Stack {
            searcher,
            syzygy,
            tt,
            ctrl,
            pos,
            nodes: None,
            replies: [const { None }; Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            value: [Default::default(); Ply::MAX as usize + 1],
            eval: [Default::default(); Ply::MAX as usize + 1],
            pv,
        }
    }

    #[inline(always)]
    fn transposition(&self) -> Option<Transposition> {
        let tpos = self.tt.get(self.pos.zobrists().hash)?;
        if tpos.best().is_none_or(|m| self.pos.is_legal(m)) {
            Some(tpos)
        } else {
            None
        }
    }

    /// The mate distance pruning.
    #[inline(always)]
    fn mdp(&self, bounds: &Range<Score>) -> (Score, Score) {
        let ply = self.pos.ply();
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    #[inline(always)]
    fn correction(&mut self) -> f32 {
        let pos = &self.pos;
        let zbs = pos.zobrists();
        let phase = pos.phase().cast::<usize>();
        let pawns = self.searcher.corrections.pawns.get(pos, zbs.pawns) as f32;
        let minor = self.searcher.corrections.minor.get(pos, zbs.minor) as f32;
        let major = self.searcher.corrections.major.get(pos, zbs.major) as f32;
        let white = self.searcher.corrections.white.get(pos, zbs.white) as f32;
        let black = self.searcher.corrections.black.get(pos, zbs.black) as f32;

        let mut correction = 0.;
        correction = Params::pawns_correction()[phase].mul_add(pawns, correction);
        correction = Params::minor_correction()[phase].mul_add(minor, correction);
        correction = Params::major_correction()[phase].mul_add(major, correction);
        correction = Params::pieces_correction()[phase].mul_add(white, correction);
        correction = Params::pieces_correction()[phase].mul_add(black, correction);
        correction / Correction::LIMIT as f32
    }

    #[inline(always)]
    fn update_correction(&mut self, depth: Depth, score: ScoreBound) {
        let pos = &self.pos;
        let ply = pos.ply();
        let zbs = pos.zobrists();
        let diff = score.bound(ply) - self.value[ply.cast::<usize>()];
        let error = diff.to_float::<f32>() * depth.get().max(1).ilog2().to_float::<f32>();

        let corrections = &mut self.searcher.corrections;
        let bonus = Params::pawns_correction_bonus()[0] * error;
        corrections.pawns.update(pos, zbs.pawns, bonus.to_int());
        let bonus = Params::minor_correction_bonus()[0] * error;
        corrections.minor.update(pos, zbs.minor, bonus.to_int());
        let bonus = Params::major_correction_bonus()[0] * error;
        corrections.major.update(pos, zbs.major, bonus.to_int());
        let bonus = Params::pieces_correction_bonus()[0] * error;
        corrections.white.update(pos, zbs.white, bonus.to_int());
        corrections.black.update(pos, zbs.black, bonus.to_int());
    }

    #[inline(always)]
    fn history_bonus(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::history_bonus_depth()),
            (1., Params::history_bonus_scalar()),
        ])
    }

    #[inline(always)]
    fn continuation_bonus(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::continuation_bonus_depth()),
            (1., Params::continuation_bonus_scalar()),
        ])
    }

    #[inline(always)]
    fn history_penalty(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::history_penalty_depth()),
            (1., Params::history_penalty_scalar()),
        ])
    }

    #[inline(always)]
    fn continuation_penalty(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::continuation_penalty_depth()),
            (1., Params::continuation_penalty_scalar()),
        ])
    }

    #[inline(always)]
    fn update_history(&mut self, depth: Depth, best: Move, moves: &Moves) {
        let pos = &self.pos;
        let ply = pos.ply();

        let bonus = Self::history_bonus(depth);
        self.searcher.history.update(pos, best, bonus.to_int());

        let bonus = Self::continuation_bonus(depth);
        let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
        reply.update(pos, best, bonus.to_int());

        for m in moves.iter() {
            if m == best {
                break;
            } else {
                let penalty = Self::history_penalty(depth);
                self.searcher.history.update(pos, m, penalty.to_int());

                let penalty = Self::continuation_penalty(depth);
                let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
                reply.update(pos, m, penalty.to_int());
            }
        }
    }

    /// A measure for how much the position is improving.
    #[inline(always)]
    fn improving(&self) -> f32 {
        if self.pos.is_check() {
            return 0.;
        }

        let ply = self.pos.ply();
        let idx = ply.cast::<usize>();
        let value = self.value[idx];

        let a = ply >= 2 && !self.pos[ply - 2].is_check() && value > self.value[idx - 2];
        let b = ply >= 4 && !self.pos[ply - 4].is_check() && value > self.value[idx - 4];

        let mut idx = Bits::<u8, 2>::new(0);
        idx.push(Bits::<u8, 1>::new(b.cast()));
        idx.push(Bits::<u8, 1>::new(a.cast()));
        *Params::improving().get(idx.cast::<usize>()).assume()
    }

    /// Computes the null move reduction.
    #[inline(always)]
    fn nmr(depth: Depth, surplus: Score) -> Option<f32> {
        match depth.get() {
            ..3 => None,
            d @ 3.. => match surplus.get() {
                ..1 => None,
                s @ 1.. => {
                    let gamma = Params::nmr_gamma()[0];
                    let delta = Params::nmr_delta()[0];
                    let limit = Params::nmr_limit()[0];
                    let flat = gamma.mul_add(s.to_float(), delta).min(limit);
                    Some(Params::nmr_fraction()[0].mul_add(d.to_float(), flat))
                }
            },
        }
    }

    /// Computes the null move pruning margin.
    #[inline(always)]
    fn nmp(depth: Depth) -> Option<f32> {
        match depth.get() {
            ..1 | 5.. => None,
            d @ 1..5 => Some(convolve([
                (d.to_float(), Params::nmp_margin_depth()),
                (1., Params::nmp_margin_scalar()),
            ])),
        }
    }

    /// Computes the fail-low pruning reduction.
    #[inline(always)]
    fn flp(depth: Depth) -> Option<f32> {
        match depth.get() {
            5.. => None,
            ..1 => Some(0.),
            d @ 1..5 => Some(convolve([
                (d.to_float(), Params::flp_margin_depth()),
                (1., Params::flp_margin_scalar()),
            ])),
        }
    }

    /// Computes fail-high pruning reduction.
    #[inline(always)]
    fn fhp(depth: Depth) -> Option<f32> {
        match depth.get() {
            7.. => None,
            ..1 => Some(0.),
            d @ 1..7 => Some(convolve([
                (d.to_float(), Params::fhp_margin_depth()),
                (1., Params::fhp_margin_scalar()),
            ])),
        }
    }

    /// Computes the razoring margin.
    #[inline(always)]
    fn razoring(depth: Depth) -> Option<f32> {
        match depth.get() {
            ..1 | 5.. => None,
            d @ 1..5 => Some(convolve([
                (d.to_float(), Params::razoring_depth()),
                (1., Params::razoring_scalar()),
            ])),
        }
    }

    /// Computes the reverse futility margin.
    #[inline(always)]
    fn rfp(depth: Depth) -> Option<f32> {
        match depth.get() {
            ..1 | 9.. => None,
            d @ 1..9 => Some(convolve([
                (d.to_float(), Params::rfp_margin_depth()),
                (1., Params::rfp_margin_scalar()),
            ])),
        }
    }

    /// Computes the futility margin.
    #[inline(always)]
    fn futility(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::fut_margin_depth()),
            (1., Params::fut_margin_scalar()),
        ])
    }

    /// Computes the probcut margin.
    #[inline(always)]
    fn probcut(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::probcut_margin_depth()),
            (1., Params::probcut_margin_scalar()),
        ])
    }

    /// Computes the singular extension margin.
    #[inline(always)]
    fn single(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::single_extension_margin_depth()),
            (1., Params::single_extension_margin_scalar()),
        ])
    }

    /// Computes the double extension margin.
    #[inline(always)]
    fn double(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::double_extension_margin_depth()),
            (1., Params::double_extension_margin_scalar()),
        ])
    }

    /// Computes the triple extension margin.
    #[inline(always)]
    fn triple(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::triple_extension_margin_depth()),
            (1., Params::triple_extension_margin_scalar()),
        ])
    }

    /// Computes the noisy SEE pruning margin.
    #[inline(always)]
    fn nsp(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::nsp_margin_depth()),
            (1., Params::nsp_margin_scalar()),
        ])
    }

    /// Computes the quiet SEE pruning margin.
    #[inline(always)]
    fn qsp(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::qsp_margin_depth()),
            (1., Params::qsp_margin_scalar()),
        ])
    }

    /// Computes the late move pruning threshold.
    #[inline(always)]
    fn lmp(depth: Depth) -> f32 {
        convolve([
            (depth.to_float(), Params::lmp_depth()),
            (1., Params::lmp_scalar()),
        ])
    }

    /// Computes the late move reduction.
    #[inline(always)]
    fn lmr(depth: Depth, index: usize) -> f32 {
        let log_depth = depth.get().max(1).ilog2();
        let log_index = index.max(1).ilog2();

        convolve([
            (log_depth.to_float(), Params::lmr_depth()),
            (log_index.to_float(), Params::lmr_index()),
            (1., Params::lmr_scalar()),
        ])
    }

    #[must_use]
    fn next(&mut self, m: Option<Move>) -> StackGuard<'_, 'a> {
        self.replies[self.pos.ply().cast::<usize>()] = m.map(|m| {
            let reply = self.searcher.continuation.reply(&self.pos, m);
            NonNull::from_mut(reply)
        });

        self.pos.push(m);
        self.tt.prefetch(self.pos.zobrists().hash);

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
        if self.pos.ply() < N as i32 && depth > 0 && bounds.start + 1 < bounds.end {
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
        let ply = self.pos.ply();
        if self.ctrl.check(depth, ply, &self.pv) == Abort {
            return Err(Interrupted);
        }

        let (alpha, beta) = match self.pos.outcome() {
            None => self.mdp(&bounds),
            Some(o) if o.is_draw() => return Ok(Pv::empty(Score::new(0))),
            Some(_) => return Ok(Pv::empty(Score::mated(ply))),
        };

        if alpha >= beta {
            return Ok(Pv::empty(alpha));
        }

        let correction = self.correction().to_int::<i16>();
        self.eval[ply.cast::<usize>()] = self.pos.evaluate();
        self.value[ply.cast::<usize>()] = self.eval[ply.cast::<usize>()] + correction;

        let is_check = self.pos.is_check();
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

        if !is_pv && self.pos.halfmoves() as f32 <= Params::tt_cut_halfmove_limit()[0] {
            if let Some(t) = transposition {
                let (lower, upper) = t.score().range(ply).into_inner();

                if let Some(margin) = Self::flp(depth - t.depth()) {
                    if upper + margin.to_int::<i16>() <= alpha {
                        return Ok(transposed.truncate());
                    }
                }

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
            match self.syzygy.wdl_after_zeroing(&self.pos) {
                None => (Score::lower(), Score::upper()),
                Some(wdl) => {
                    let bounds = Score::losing(Ply::upper())..Score::winning(Ply::upper());
                    let score = ScoreBound::new(bounds, wdl.to_score(ply), ply);
                    let (lower, upper) = score.range(ply).into_inner();
                    if lower >= upper || upper <= alpha || lower >= beta {
                        let bonus = Params::tb_cut_depth_bonus()[0].to_int::<i8>();
                        let tpos = Transposition::new(score, depth + bonus, None, was_pv);
                        self.tt.set(self.pos.zobrists().hash, tpos);
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
        } else if !is_pv && !is_check && depth > 0 {
            if let Some(margin) = Self::razoring(depth) {
                if self.value[ply.cast::<usize>()] + margin.to_int::<i16>() <= alpha {
                    let pv = self.nw(Depth::new(0), beta, cut)?;
                    if pv <= alpha {
                        return Ok(pv);
                    }
                }
            }

            if let Some(mut margin) = Self::rfp(depth) {
                margin = Params::rfp_margin_improving()[0].mul_add(improving, margin);
                if transposed.score() - margin.to_int::<i16>() >= beta {
                    return Ok(transposed.truncate());
                }
            }

            let turn = self.pos.turn();
            let pawns = self.pos.pawns(turn);
            if (self.pos.material(turn) ^ pawns).len() > 1 {
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

        let move_pack = self.pos.moves();
        let mut moves = Moves::from_iter(move_pack.unpack_if(|ms| depth > 0 || !ms.is_quiet()));
        let killer = self.killers[ply.cast::<usize>()];

        moves.sort(|m| {
            if Some(m) == transposed.head() {
                return Bounded::upper();
            }

            let pos = &self.pos;
            let mut rating = killer.contains(m).to_float::<f32>() * Params::killer_rating()[0];

            let history = self.searcher.history.get(pos, m).to_float::<f32>();
            rating = Params::history_rating()[0].mul_add(history / History::LIMIT as f32, rating);

            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let counter = reply.get(pos, m).to_float::<f32>();
            rating = Params::counter_rating()[0].mul_add(counter / History::LIMIT as f32, rating);

            if !m.is_quiet() {
                if pos.winning(m, Params::winning_rating_margin()[0].to_int()) {
                    rating += convolve([
                        (pos.gain(m).to_float(), Params::winning_rating_gain()),
                        (1., Params::winning_rating_scalar()),
                    ]);
                }
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

                    let margin = p_beta - self.value[ply.cast::<usize>()];
                    if !self.pos.winning(m, margin.saturate()) {
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
                        self.tt.set(self.pos.zobrists().hash, tpos);
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

            let mut lmp = Params::lmp_baseline()[0];
            lmp = Params::lmp_is_pv()[0].mul_add(is_pv.to_float(), lmp);
            lmp = Params::lmp_was_pv()[0].mul_add(was_pv.to_float(), lmp);
            lmp = Params::lmp_is_check()[0].mul_add(is_check.to_float(), lmp);
            lmp = Params::lmp_improving()[0].mul_add(improving, lmp);
            if index.to_float::<f32>() > Self::lmp(depth) * lmp {
                break;
            }

            let pos = &self.pos;
            let mut lmr = Self::lmr(depth, index);
            let lmr_depth = depth - lmr.to_int::<i8>();
            let history = self.searcher.history.get(pos, m).to_float::<f32>();
            let mut reply = self.replies.get_mut(ply.cast::<usize>().wrapping_sub(1));
            let counter = reply.get(pos, m).to_float::<f32>();
            let gain = pos.gain(m).to_float::<f32>();
            let is_killer = killer.contains(m);
            let is_quiet = m.is_quiet();

            let mut fut = Self::futility(lmr_depth);
            fut = Params::fut_margin_is_pv()[0].mul_add(is_pv.to_float(), fut);
            fut = Params::fut_margin_was_pv()[0].mul_add(was_pv.to_float(), fut);
            fut = Params::fut_margin_is_check()[0].mul_add(is_check.to_float(), fut);
            fut = Params::fut_margin_is_killer()[0].mul_add(is_killer.to_float(), fut);
            fut = Params::fut_margin_improving()[0].mul_add(improving, fut);
            fut = Params::fut_margin_gain()[0].mul_add(gain, fut);
            if self.value[ply.cast::<usize>()] + fut.to_int::<i16>().max(0) <= alpha {
                continue;
            }

            let mut spt = if is_quiet {
                Self::qsp(lmr_depth)
            } else {
                Self::nsp(depth)
            };

            spt = Params::sp_margin_is_killer()[0].mul_add(is_killer.to_float(), spt);
            if !pos.winning(m, spt.to_int()) {
                continue;
            }

            let mut next = self.next(Some(m));
            let gives_check = next.pos.is_check();

            lmr += Params::lmr_baseline()[0];
            lmr = Params::lmr_is_pv()[0].mul_add(is_pv.to_float(), lmr);
            lmr = Params::lmr_was_pv()[0].mul_add(was_pv.to_float(), lmr);
            lmr = Params::lmr_gives_check()[0].mul_add(gives_check.to_float(), lmr);
            lmr = Params::lmr_is_noisy_pv()[0].mul_add(is_noisy_pv.to_float(), lmr);
            lmr = Params::lmr_is_killer()[0].mul_add(is_killer.to_float(), lmr);
            lmr = Params::lmr_cut()[0].mul_add(cut.to_float(), lmr);
            lmr = Params::lmr_improving()[0].mul_add(improving, lmr);
            lmr = Params::lmr_history()[0].mul_add(history / History::LIMIT as f32, lmr);
            lmr = Params::lmr_counter()[0].mul_add(counter / History::LIMIT as f32, lmr);

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
        self.tt.set(self.pos.zobrists().hash, tpos);

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
        if self.ctrl.check(depth, Ply::new(0), &self.pv) != Continue {
            return Err(Interrupted);
        }

        let is_check = self.pos.is_check();
        let is_noisy_pv = self.pv.head().is_some_and(|m| !m.is_quiet());
        self.value[0] = self.eval[0] + self.correction().to_int::<i16>();

        moves.sort(|m| {
            if Some(m) == self.pv.head() {
                return Bounded::upper();
            }

            let mut rating = 0.;
            let pos = &self.pos;
            let history = self.searcher.history.get(pos, m).to_float::<f32>();
            rating = Params::history_rating()[0].mul_add(history / History::LIMIT as f32, rating);

            if !m.is_quiet() {
                if pos.winning(m, Params::winning_rating_margin()[0].to_int()) {
                    rating += convolve([
                        (pos.gain(m).to_float(), Params::winning_rating_gain()),
                        (1., Params::winning_rating_scalar()),
                    ]);
                }
            }

            rating.to_int()
        });

        let mut sorted_moves = moves.sorted();
        let mut head = sorted_moves.next().assume();
        self.nodes = self.ctrl.attention(head);
        let mut tail = -self.next(Some(head)).ab(depth - 1, -beta..-alpha, false)?;

        let mut is_noisy_node = is_check || is_noisy_pv;
        is_noisy_node |= !head.is_quiet() && tail > alpha;
        for (index, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut lmp = Params::lmp_is_root()[0];
            lmp = Params::lmp_is_check()[0].mul_add(is_check.to_float(), lmp);
            if index.to_float::<f32>() > Self::lmp(depth) * lmp {
                break;
            }

            let history = self.searcher.history.get(&self.pos, m).to_float::<f32>();
            self.nodes = self.ctrl.attention(m);

            let mut next = self.next(Some(m));
            let gives_check = next.pos.is_check();

            let mut lmr = Self::lmr(depth, index);
            lmr += Params::lmr_is_root()[0];
            lmr = Params::lmr_gives_check()[0].mul_add(gives_check.to_float(), lmr);
            lmr = Params::lmr_is_noisy_pv()[0].mul_add(is_noisy_pv.to_float(), lmr);
            lmr = Params::lmr_history()[0].mul_add(history / History::LIMIT as f32, lmr);

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
        self.tt.set(self.pos.zobrists().hash, tpos);

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
    fn aw(&mut self) -> impl Iterator<Item = (Depth, Pv)> {
        gen move {
            let clock = self.ctrl.limits().clock;
            let mut moves = Moves::from_iter(self.pos.moves().unpack());
            let mut stop = matches!((moves.len(), &clock), (0, _) | (1, Some(_)));
            let mut depth = Depth::new(0);

            self.eval[0] = self.pos.evaluate();
            self.value[0] = self.eval[0] + self.correction().to_int::<i16>();
            if let Some(t) = self.transposition().filter(|t| t.best().is_some()) {
                self.pv = t.transpose(Ply::new(0)).truncate();
            } else if let Some(m) = moves.iter().next() {
                self.pv = Pv::new(self.value[0].saturate(), Line::singular(m));
            }

            loop {
                yield (depth, self.pv.clone().truncate());
                stop |= self.ctrl.check(depth + 1, Ply::new(0), &self.pv) != Continue;
                if stop || depth >= Depth::upper() {
                    return;
                }

                depth += 1;
                let mut reduction = 0.;
                let mut window = Params::aw_baseline()[0];
                let (mut lower, mut upper) = match depth.get() {
                    ..=4 => (Score::lower(), Score::upper()),
                    _ => (
                        self.pv.score() - window.to_int::<i16>(),
                        self.pv.score() + window.to_int::<i16>(),
                    ),
                };

                loop {
                    let depth = depth - reduction.to_int::<i8>();
                    window = window.mul_add(Params::aw_gamma()[0], Params::aw_delta()[0]);
                    let partial = match self.root(&mut moves, depth, lower..upper) {
                        Err(_) => break stop = true,
                        Ok(pv) => pv,
                    };

                    match partial.score() {
                        score if (-lower..Score::upper()).contains(&-score) => {
                            let blend = Params::aw_fail_low_blend()[0];
                            upper = blend.lerp(lower.to_float(), upper.to_float()).to_int();
                            lower = score - window.to_int::<i16>();
                            reduction = 0.;
                        }

                        score if (upper..Score::upper()).contains(&score) => {
                            upper = score + window.to_int::<i16>();
                            reduction += Params::aw_fail_high_reduction()[0];
                            self.pv = partial;
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
pub struct Search<'e> {
    engine: &'e mut Engine,
    pos: Evaluator,
    ctrl: GlobalControl,
    channel: Option<UnboundedReceiver<Info>>,
    task: Option<Task<'e>>,
}

impl<'e> Search<'e> {
    fn new(engine: &'e mut Engine, pos: &Position, limits: Limits) -> Self {
        Search {
            engine,
            pos: Evaluator::new(pos.clone()),
            ctrl: GlobalControl::new(pos, limits),
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

impl<'e> Drop for Search<'e> {
    fn drop(&mut self) {
        if let Some(t) = self.task.take() {
            self.abort();
            drop(t);
        }
    }
}

impl<'e> FusedStream for Pin<&mut Search<'e>> {
    fn is_terminated(&self) -> bool {
        self.channel
            .as_ref()
            .is_some_and(FusedStream::is_terminated)
    }
}

impl<'e> Stream for Pin<&mut Search<'e>> {
    type Item = Info;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(rx) = &mut self.channel {
            return rx.poll_next_unpin(cx);
        }

        let executor: &mut Executor = unsafe { &mut *(&mut self.engine.executor as *mut _) };
        let syzygy: &Syzygy = unsafe { &*(&self.engine.syzygy as *const _) };
        let tt: &Memory<Transposition> = unsafe { &*(&self.engine.tt as *const _) };
        let ctrl: &GlobalControl = unsafe { &*(&self.ctrl as *const _) };
        let pos: &Evaluator = unsafe { &*(&self.pos as *const _) };
        let searchers: &[SyncUnsafeCell<Searcher>] =
            unsafe { &*(&mut *self.engine.searchers as *mut _ as *const _) };

        let (tx, rx) = unbounded();
        self.channel = Some(rx);
        if let Some(pv) = syzygy.best(pos) {
            let info = Info::new(Depth::new(0), Duration::ZERO, 0, pv.truncate());
            return Poll::Ready(Some(info));
        }

        self.task = Some(executor.execute(move |idx| {
            let local = if idx == 0 {
                LocalControl::active(ctrl)
            } else {
                LocalControl::passive(ctrl)
            };

            let searcher = unsafe { &mut *searchers.get(idx).assume().get() };
            for (depth, pv) in Stack::new(searcher, syzygy, tt, local, pos.clone()).aw() {
                if idx == 0 {
                    let info = Info::new(depth, ctrl.elapsed(), ctrl.visited(), pv);
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
    searchers: Slice<Searcher>,
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
            searchers: Slice::new(options.threads.cast()).unwrap(),
        }
    }

    /// Resets the hash size.
    pub fn set_hash(&mut self, hash: HashSize) {
        let mut to_drop = Memory::new(0);
        swap(&mut self.tt, &mut to_drop);
        drop(to_drop); // IMPORTANT: deallocate before reallocating
        self.tt = Memory::new(hash.get());
    }

    /// Resets the thread count.
    pub fn set_threads(&mut self, threads: ThreadCount) {
        self.executor = Executor::new(threads);
        self.searchers = Slice::new(threads.cast()).unwrap();
    }

    /// Resets the Syzygy path.
    pub fn set_syzygy<I: IntoIterator<Item: AsRef<Path>>>(&mut self, paths: I) {
        self.syzygy = Syzygy::new(paths);
    }

    /// Resets the engine state.
    pub fn reset(&mut self) {
        let tt: &[SyncUnsafeCell<Vault<Transposition>>] =
            unsafe { &*(&mut *self.tt as *mut _ as *const _) };
        let searchers: &[SyncUnsafeCell<Searcher>] =
            unsafe { &*(&mut *self.searchers as *mut _ as *const _) };

        let tt_chunk_size = tt.len().div_ceil(searchers.len());
        self.executor.execute(move |idx| unsafe {
            let offset = idx * tt_chunk_size;
            let len = tt.len().saturating_sub(offset).min(tt_chunk_size);
            let ptr = tt.as_ptr() as *mut Vault<Transposition>;
            fill_zeroes(slice::from_raw_parts_mut(ptr.add(offset), len));
            *searchers.get(idx).assume().get() = zeroed();
        });
    }

    /// Initiates a [`Search`].
    pub fn search(&mut self, pos: &Position, limits: Limits) -> Search<'_> {
        Search::new(self, pos, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{executor::block_on_stream, pin_mut};
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
        #[filter(!#b.is_winning() && !#b.is_losing())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_winning() && #s >= #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), Depth::upper(), Some(m), was_pv);
        e.tt.set(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);
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
        #[filter(!#b.is_winning() && !#b.is_losing())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_losing() && #s < #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), Depth::upper(), Some(m), was_pv);
        e.tt.set(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);
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
        #[filter(!#b.is_winning() && !#b.is_losing())] b: Score,
        was_pv: bool,
        d: Depth,
        #[filter(!#s.is_winning() && !#s.is_losing())] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), Depth::upper(), Some(m), was_pv);
        e.tt.set(pos.zobrists().hash, tpos);

        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
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
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);
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
        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);
        stack.pv = stack.pv.transpose(m);
        global.abort();
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
        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);

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
        let global = GlobalControl::new(&pos, Limits::none());
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        stack.nodes = stack.ctrl.attention(m);

        assert_eq!(stack.ab::<1>(d, b, cut), Ok(Pv::empty(Score::mated(ply))));
    }

    #[proptest]
    fn aw_extends_time_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let pos = Evaluator::new(pos.clone());
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|(_, pv)| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_depth_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let pos = Evaluator::new(pos.clone());
        let global = GlobalControl::new(&pos, Limits::depth(Depth::lower()));
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|(_, pv)| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_nodes_to_find_some_pv(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let pos = Evaluator::new(pos.clone());
        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let ctrl = LocalControl::active(&global);
        let mut stack = Stack::new(&mut e.searchers[0], &e.syzygy, &e.tt, ctrl, pos);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|(_, pv)| pv.head()), None);
    }

    #[proptest]
    fn search_returns_pvs_that_improve_monotonically(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
        d: Depth,
    ) {
        let search = e.search(&pos, Limits::depth(d));
        pin_mut!(search);

        let infos = block_on_stream(search);
        assert!(infos.map(|i| (i.depth(), i.score())).is_sorted());
    }
}
