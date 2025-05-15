use crate::chess::{Move, Moves, Position};
use crate::nnue::{Evaluator, Value};
use crate::util::{Assume, Integer};
use crate::{params::Params, search::*};
use derive_more::with_trait::{Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{ops::Range, pin::Pin};

#[cfg(test)]
use proptest::prelude::*;

/// Indicates the search was interrupted .
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Error)]
#[display("the search was interrupted")]
pub struct Interrupted;

#[derive(Debug)]
struct Stack<'a> {
    searcher: &'a Searcher,
    tt: &'a TranspositionTable,
    ctrl: &'a Control,
    root: &'a Evaluator,
    nodes: Option<&'a Counter>,
    replies: [Option<&'a Reply>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
    pv: Pv,
}

impl<'a> Stack<'a> {
    fn new(
        searcher: &'a Searcher,
        tt: &'a TranspositionTable,
        ctrl: &'a Control,
        root: &'a Evaluator,
    ) -> Self {
        Stack {
            searcher,
            tt,
            ctrl,
            root,
            nodes: None,
            replies: [Default::default(); Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            value: [Default::default(); Ply::MAX as usize + 1],
            pv: Pv::empty(Score::lower()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        pos: &Position,
        moves: &Moves<Value>,
        bounds: Range<Score>,
        depth: Depth,
        ply: Ply,
        best: Move,
        score: Score,
    ) {
        let draft = depth - ply;
        if score >= bounds.end {
            if best.is_quiet() {
                self.killers[ply.cast::<usize>()].insert(best);
            }

            let bonus_alpha: i32 = Params::history_bonus_alpha().as_int();
            let bonus_beta: i32 = Params::history_bonus_beta().as_int();
            let bonus_scale: i32 = Params::history_bonus_scale().as_int();
            let bonus = (draft.cast::<i32>() * bonus_alpha + bonus_beta) / bonus_scale;

            let penalty_alpha: i32 = Params::history_penalty_alpha().as_int();
            let penalty_beta: i32 = Params::history_penalty_beta().as_int();
            let penalty_scale: i32 = Params::history_penalty_scale().as_int();
            let penalty = -(draft.cast::<i32>() * penalty_alpha + penalty_beta) / penalty_scale;

            self.searcher.history.update(pos, best, bonus.saturate());

            let counter = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
            counter.update(pos, best, bonus.saturate());

            for &(m, _) in moves.iter().rev() {
                if m == best {
                    break;
                } else {
                    self.searcher.history.update(pos, m, penalty.saturate());

                    let counter = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
                    counter.update(pos, m, penalty.saturate());
                }
            }
        }

        let score = ScoreBound::new(bounds, score, ply);
        let tpos = Transposition::new(score, draft, best);
        self.tt.set(pos.zobrist(), tpos);
    }

    /// A measure for how much the position is improving.
    fn improving(&mut self, ply: Ply) -> i32 {
        let idx = ply.cast::<usize>();

        let a = (idx >= 2 && self.value[idx] > self.value[idx - 2]) as i32;
        let b = (idx >= 4 && self.value[idx] > self.value[idx - 4]) as i32;

        a + b
    }

    /// The mate distance pruning.
    fn mdp(&self, ply: Ply, bounds: &Range<Score>) -> (Score, Score) {
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    /// Computes the null move pruning reduction.
    fn nmp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        let alpha: i32 = Params::null_move_reduction_alpha().as_int();
        let beta: i32 = Params::null_move_reduction_beta().as_int();
        let scale: i32 = Params::value_scale().as_int();

        match scale * surplus.cast::<i32>() {
            ..0 => None,
            s if s >= 3 * alpha - beta => Some(draft - 3 - draft / 4),
            s => Some(draft - (s + beta) / alpha - draft / 4),
        }
    }

    /// Computes fail-high pruning reduction.
    fn fhp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        let alpha: i32 = Params::fail_high_reduction_alpha().as_int();
        let beta: i32 = Params::fail_high_reduction_beta().as_int();
        let scale: i32 = Params::value_scale().as_int();

        match scale * surplus.cast::<i32>() {
            ..0 => None,
            s if s >= 3 * alpha - beta => Some(draft - 3),
            s => Some(draft - (s + beta) / alpha),
        }
    }

    /// Computes the fail-low pruning reduction.
    fn flp(&self, deficit: Score, draft: Depth) -> Option<Depth> {
        let alpha: i32 = Params::fail_low_reduction_alpha().as_int();
        let beta: i32 = Params::fail_low_reduction_beta().as_int();
        let scale: i32 = Params::value_scale().as_int();

        match scale * deficit.cast::<i32>() {
            ..0 => None,
            s if s >= 3 * alpha - beta => Some(draft - 3),
            s => Some(draft - (s + beta) / alpha),
        }
    }

    /// Computes the singular extension margin.
    fn singular(&self, draft: Depth) -> i32 {
        let alpha: i32 = Params::singular_extension_margin_alpha().as_int();
        let beta: i32 = Params::singular_extension_margin_beta().as_int();
        let scale: i32 = Params::value_scale().as_int();

        (alpha * draft.cast::<i32>() + beta) / scale
    }

    /// Computes the futility margin.
    fn futility(&self, draft: Depth) -> i32 {
        let alpha: i32 = Params::futility_margin_alpha().as_int();
        let beta: i32 = Params::futility_margin_beta().as_int();
        let scale: i32 = Params::value_scale().as_int();

        (alpha * draft.cast::<i32>() + beta) / scale
    }

    /// Computes the futility pruning threshold.
    fn fpt(&self, draft: Depth) -> i32 {
        let alpha: i32 = Params::futility_pruning_threshold_alpha().as_int();
        let scale: i32 = Params::value_scale().as_int();

        alpha * draft.cast::<i32>() / scale
    }

    /// Computes the SEE pruning threshold.
    fn spt(&self, draft: Depth) -> i32 {
        let alpha: i32 = Params::see_pruning_threshold_alpha().as_int();
        let scale: i32 = Params::value_scale().as_int();

        alpha * draft.cast::<i32>() / scale
    }

    /// Computes the late move reduction.
    fn lmr(&self, draft: Depth, idx: usize) -> i32 {
        let alpha: i32 = Params::late_move_reduction_alpha().as_int();
        let beta: i32 = Params::late_move_reduction_beta().as_int();
        let scale: i32 = Params::late_move_reduction_scale().as_int();

        let x = idx.max(1).ilog2() as i32;
        let y = draft.get().max(1).ilog2() as i32;
        (x * y * alpha + beta) / scale
    }

    /// Computes the late move pruning threshold.
    fn lmp(&self, draft: Depth, idx: usize) -> i32 {
        let alpha: i32 = Params::late_move_pruning_alpha().as_int();
        let beta: i32 = Params::late_move_pruning_beta().as_int();
        let scale: i32 = Params::late_move_pruning_scale().as_int();

        scale * idx.cast::<i32>() / (beta + alpha * draft.cast::<i32>().pow(2))
    }

    /// The zero-window alpha-beta search.
    fn nw<const N: usize>(
        &mut self,
        pos: &Evaluator,
        beta: Score,
        depth: Depth,
        ply: Ply,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        self.ab(pos, beta - 1..beta, depth, ply, cut)
    }

    /// The alpha-beta search.
    fn ab<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        depth: Depth,
        ply: Ply,
        cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        if ply.cast::<usize>() < N && depth > ply && bounds.start + 1 < bounds.end {
            self.pvs(pos, bounds, depth, ply, cut)
        } else {
            Ok(self.pvs::<0>(pos, bounds, depth, ply, cut)?.truncate())
        }
    }

    /// The principal variation search.
    fn pvs<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        mut depth: Depth,
        ply: Ply,
        mut cut: bool,
    ) -> Result<Pv<N>, Interrupted> {
        (ply > 0).assume();
        self.nodes.update(1);
        if self.ctrl.check(self.root, &self.pv, ply) == ControlFlow::Abort {
            return Err(Interrupted);
        }

        let (alpha, beta) = match pos.outcome() {
            None => self.mdp(ply, &bounds),
            Some(o) if o.is_draw() => return Ok(Pv::empty(Score::new(0))),
            Some(_) => return Ok(Pv::empty(Score::mated(ply))),
        };

        if alpha >= beta {
            return Ok(Pv::empty(alpha));
        }

        self.value[ply.cast::<usize>()] = pos.evaluate();
        let transposition = self.tt.get(pos.zobrist());
        let transposed = match transposition {
            None => Pv::empty(self.value[ply.cast::<usize>()].saturate()),
            Some(t) => t.transpose(ply),
        };

        depth += pos.is_check() as i8;
        depth -= transposition.is_none() as i8;

        let draft = depth - ply;
        let quiesce = draft <= 0;
        let is_pv = alpha + 1 < beta;
        if let Some(t) = transposition {
            let (lower, upper) = t.score().range(ply).into_inner();

            if lower >= upper || upper <= alpha || lower >= beta {
                if !is_pv && t.draft() >= draft {
                    return Ok(transposed.truncate());
                }
            }

            if let Some(d) = self.fhp(lower - beta, draft) {
                if !is_pv && t.draft() >= d {
                    return Ok(transposed.truncate());
                }
            }

            if let Some(d) = self.flp(alpha - upper, draft) {
                if !is_pv && t.draft() >= d {
                    return Ok(transposed.truncate());
                }
            }
        }

        let alpha = if quiesce {
            transposed.score().max(alpha)
        } else {
            alpha
        };

        if alpha >= beta || ply >= Ply::MAX {
            return Ok(transposed.truncate());
        } else if let Some(d) = self.nmp(transposed.score() - beta, draft) {
            if !is_pv && !pos.is_check() && pos.pieces(pos.turn()).len() > 1 {
                if d <= 0 {
                    return Ok(transposed.truncate());
                } else {
                    let mut next = pos.clone();
                    next.pass();
                    self.tt.prefetch(next.zobrist());
                    self.replies[ply.cast::<usize>()] = None;
                    if -self.nw::<0>(&next, -beta + 1, d + ply, ply + 1, !cut)? >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let value_scale: i16 = Params::value_scale().as_int();
        let killer_bonus: i16 = Params::killer_move_bonus().as_int();
        let killer = self.killers[ply.cast::<usize>()];
        let mut moves: Moves<_> = pos
            .moves()
            .unpack_if(|ms| !quiesce || !ms.is_quiet())
            .map(|m| {
                if Some(m) == transposed.head() {
                    return (m, Value::upper());
                }

                let history = &self.searcher.history;
                let counter = self.replies[ply.cast::<usize>() - 1];
                let mut rating = pos.gain(m) + history.get(pos, m) + counter.get(pos, m);

                if killer.contains(m) {
                    rating += killer_bonus / value_scale;
                }

                (m, rating)
            })
            .collect();

        moves.sort_unstable_by_key(|(_, rating)| *rating);
        let (mut head, mut tail) = match moves.last() {
            None => return Ok(transposed.truncate()),
            Some(&(m, _)) => {
                let mut extension = 0i8;
                if let Some(t) = transposition {
                    if t.score().lower(ply) >= beta && t.draft() >= draft - 3 && draft >= 6 {
                        extension = 1;
                        let s_draft = (draft - 1) / 2;
                        let s_beta = beta - self.singular(draft);
                        for (m, _) in moves.iter().rev().skip(1) {
                            let mut next = pos.clone();
                            next.play(*m);
                            self.tt.prefetch(next.zobrist());
                            self.replies[ply.cast::<usize>()] =
                                Some(self.searcher.continuation.reply(pos, *m));
                            let pv = -self.nw(&next, -s_beta + 1, s_draft + ply, ply + 1, !cut)?;
                            if pv >= beta {
                                return Ok(pv.transpose(*m));
                            } else if pv >= s_beta {
                                cut = true;
                                extension = -1;
                                break;
                            }
                        }
                    }
                }

                let mut next = pos.clone();
                next.play(m);
                self.tt.prefetch(next.zobrist());
                self.replies[ply.cast::<usize>()] = Some(self.searcher.continuation.reply(pos, m));
                let pv = -self.ab(&next, -beta..-alpha, depth + extension, ply + 1, false)?;
                (m, pv)
            }
        };

        let improving = self.improving(ply);
        for (idx, &(m, _)) in moves.iter().rev().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if self.lmp(draft, idx) > improving {
                break;
            }

            let lmr = Depth::new(cut as _) + self.lmr(draft, idx) - is_pv as i8 - improving;
            if self.value[ply.cast::<usize>()] + self.futility(draft - lmr) <= alpha {
                let threshold = self.fpt(draft - lmr);
                if !pos.winning(m, Value::new(1) + threshold) {
                    continue;
                }
            }

            if !pos.winning(m, Value::new(1) - self.spt(draft - lmr)) {
                continue;
            }

            let mut next = pos.clone();
            next.play(m);
            self.tt.prefetch(next.zobrist());
            self.replies[ply.cast::<usize>()] = Some(self.searcher.continuation.reply(pos, m));
            let pv = match -self.nw(&next, -alpha, depth - lmr, ply + 1, !cut || lmr > 0)? {
                pv if pv <= alpha || (pv >= beta && lmr <= 0) => pv,
                _ => -self.ab(&next, -beta..-alpha, depth, ply + 1, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        self.record(pos, &moves, bounds, depth, ply, head, tail.score());
        Ok(tail.transpose(head))
    }

    /// The root of the principal variation search.
    fn root(
        &mut self,
        moves: &mut Moves<Value>,
        bounds: Range<Score>,
        depth: Depth,
    ) -> Result<Pv, Interrupted> {
        let ply = Ply::new(0);
        let (alpha, beta) = (bounds.start, bounds.end);
        if self.ctrl.check(self.root, &self.pv, ply) != ControlFlow::Continue {
            return Err(Interrupted);
        }

        for (m, rating) in moves.iter_mut() {
            if Some(*m) == self.pv.head() {
                *rating = Value::upper();
            } else {
                *rating = self.root.gain(*m) + self.searcher.history.get(self.root, *m);
            }
        }

        moves.sort_unstable_by_key(|(_, rating)| *rating);
        let &(mut head, _) = moves.last().assume();
        let mut next = self.root.clone();
        next.play(head);
        self.tt.prefetch(next.zobrist());
        self.nodes = Some(self.ctrl.attention().nodes(head));
        self.replies[0] = Some(self.searcher.continuation.reply(self.root, head));
        let mut tail = -self.ab(&next, -beta..-alpha, depth, ply + 1, false)?;

        for (idx, &(m, _)) in moves.iter().rev().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if self.lmp(depth, idx) > 0 {
                break;
            }

            let lmr = Depth::new(0) + self.lmr(depth, idx);
            if self.value[0] + self.futility(depth - lmr) <= alpha {
                let threshold = self.fpt(depth - lmr);
                if !self.root.winning(m, Value::new(1) + threshold) {
                    continue;
                }
            }

            let mut next = self.root.clone();
            next.play(m);
            self.tt.prefetch(next.zobrist());
            self.nodes = Some(self.ctrl.attention().nodes(m));
            self.replies[0] = Some(self.searcher.continuation.reply(self.root, m));
            let pv = match -self.nw(&next, -alpha, depth - lmr, ply + 1, false)? {
                pv if pv <= alpha || (pv >= beta && lmr <= 0) => pv,
                _ => -self.ab(&next, -beta..-alpha, depth, ply + 1, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        self.record(self.root, moves, bounds, depth, ply, head, tail.score());
        Ok(tail.transpose(head))
    }

    /// An implementation of aspiration windows with iterative deepening.
    fn aw(&mut self) -> impl Iterator<Item = Info> {
        gen move {
            let pos = self.root;
            let mut depth = Depth::new(0);
            let mut moves: Moves<_> = pos.moves().unpack().map(|m| (m, pos.gain(m))).collect();
            self.value[0] = pos.evaluate();
            self.pv = match moves.iter().max_by_key(|(_, rating)| *rating) {
                None if !pos.is_check() => Pv::empty(Score::new(0)),
                None => Pv::empty(Score::mated(Ply::new(0))),
                Some((m, _)) => match self.tt.get(pos.zobrist()) {
                    None => Pv::new(self.value[0].saturate(), Line::singular(*m)),
                    Some(t) => t.transpose(Ply::new(0)).truncate(),
                },
            };

            let mut stop = false;
            let limits = self.ctrl.limits();
            if matches!((&*moves, limits), (&[], _) | (&[_], Limits::Clock(..))) {
                stop = true;
            }

            let value_scale: i32 = Params::value_scale().as_int();
            let aw_start: i32 = Params::aspiration_window_start().as_int();
            let aw_alpha: i32 = Params::aspiration_window_alpha().as_int();
            let aw_beta: i32 = Params::aspiration_window_beta().as_int();

            loop {
                let pv = self.pv.clone().truncate();
                yield Info::new(depth, self.ctrl.time(), self.ctrl.nodes(), pv);
                if stop || depth >= Depth::upper() {
                    return;
                }

                depth += 1;
                let mut draft = depth;
                let mut delta = aw_start / value_scale;
                let (mut lower, mut upper) = match depth.get() {
                    ..=4 => (Score::lower(), Score::upper()),
                    _ => (self.pv.score() - delta, self.pv.score() + delta),
                };

                loop {
                    delta = (delta * aw_alpha + aw_beta) / value_scale;
                    let partial = match self.root(&mut moves, lower..upper, draft) {
                        Err(_) => break stop = true,
                        Ok(pv) => pv,
                    };

                    match partial.score() {
                        score if (-lower..Score::upper()).contains(&-score) => {
                            draft = depth;
                            upper = lower / 2 + upper / 2;
                            lower = score - delta;
                        }

                        score if (upper..Score::upper()).contains(&score) => {
                            draft = Depth::new(1).max(draft - 1);
                            upper = score + delta;
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
pub struct Search<'e, 'c, 'p> {
    engine: &'e mut Engine,
    ctrl: &'c Control,
    position: &'p Evaluator,
    channel: Option<(UnboundedReceiver<Info>, Task<'e>)>,
}

impl<'e, 'c, 'p> Search<'e, 'c, 'p> {
    fn new(engine: &'e mut Engine, ctrl: &'c Control, position: &'p Evaluator) -> Self {
        Search {
            engine,
            ctrl,
            position,
            channel: None,
        }
    }
}

impl<'e, 'p, 'c> Drop for Search<'e, 'p, 'c> {
    fn drop(&mut self) {
        if let Some((channel, task)) = self.channel.take() {
            self.ctrl.abort();
            drop(task);
            drop(channel);
        }
    }
}

impl<'e, 'p, 'c> FusedStream for Search<'e, 'p, 'c> {
    fn is_terminated(&self) -> bool {
        self.channel
            .as_ref()
            .is_some_and(|(c, _)| c.is_terminated())
    }
}

impl<'e, 'p, 'c> Stream for Search<'e, 'p, 'c> {
    type Item = Info;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some((channel, _)) = &mut self.channel {
            return channel.poll_next_unpin(cx);
        }

        let executor: &mut Executor = unsafe { &mut *(&mut self.engine.executor as *mut _) };
        let searchers: &'static [Searcher] = unsafe { &*(&*self.engine.searchers as *const _) };
        let tt: &'static TranspositionTable = unsafe { &*(&self.engine.tt as *const _) };
        let ctrl: &'static Control = unsafe { &*(self.ctrl as *const _) };
        let root: &'static Evaluator = unsafe { &*(self.position as *const _) };

        let (tx, rx) = unbounded();
        let task = executor.execute(move |idx| {
            let searcher = searchers.get(idx).assume();
            for info in Stack::new(searcher, tt, ctrl, root).aw() {
                if idx == 0 {
                    let depth = info.depth();
                    tx.unbounded_send(info).assume();
                    if depth >= ctrl.limits().depth() {
                        break;
                    }
                }
            }

            if idx == 0 {
                tx.close_channel();
                ctrl.abort();
            }
        });

        self.channel = Some((rx, task));
        self.poll_next(cx)
    }
}

#[derive(Debug, Default)]
struct Searcher {
    history: History,
    continuation: Continuation,
}

/// A chess engine.
#[derive(Debug)]
pub struct Engine {
    tt: TranspositionTable,
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
            tt: TranspositionTable::new(options.hash),
            executor: Executor::new(options.threads),
            searchers: (0..options.threads.get())
                .map(|_| Searcher::default())
                .collect(),
        }
    }

    /// Initiates a [`Search`].
    pub fn search<'e, 'p, 'c>(
        &'e mut self,
        pos: &'p Evaluator,
        ctrl: &'c Control,
    ) -> Search<'e, 'c, 'p> {
        Search::new(self, ctrl, pos)
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
        assert!(e.tt.size() <= o.hash);
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_low(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s >= #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(&pos, b, d, p, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s < #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(&pos, b, d, p, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none())] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(&pos, b, d, p, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn ab_returns_static_evaluation_if_max_ply(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);

        assert_eq!(
            stack.ab::<1>(&pos, Score::lower()..Score::upper(), d, Ply::upper(), cut),
            Ok(Pv::empty(pos.evaluate().saturate()))
        );
    }

    #[proptest]
    fn ab_aborts_if_maximum_number_of_nodes_visited(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::Nodes(0));
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.ab::<1>(&pos, b, d, p, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_aborts_if_time_is_up(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        thread::sleep(Duration::from_millis(1));
        assert_eq!(stack.ab::<1>(&pos, b, d, p, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_can_be_aborted_upon_request(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        ctrl.abort();
        assert_eq!(stack.ab::<1>(&pos, b, d, p, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        #[by_ref] e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);

        assert_eq!(
            stack.ab::<1>(&pos, b, d, p, cut),
            Ok(Pv::empty(Score::new(0)))
        );
    }

    #[proptest]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);

        assert_eq!(
            stack.ab::<1>(&pos, b, d, p, cut),
            Ok(Pv::empty(Score::mated(p)))
        );
    }

    #[proptest]
    fn aw_extends_time_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_depth_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let ctrl = Control::new(&pos, Limits::Depth(Depth::lower()));
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn search_returns_pvs_that_improve_monotonically(
        mut e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        d: Depth,
    ) {
        let ctrl = Control::new(&pos, Limits::Depth(d));
        let infos = block_on_stream(e.search(&pos, &ctrl));
        assert!(infos.map(|i| (i.depth(), i.score())).is_sorted());
    }
}
