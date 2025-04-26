use crate::chess::{Move, Position};
use crate::nnue::{Evaluator, Value};
use crate::search::*;
use crate::util::{Assume, Integer};
use arrayvec::ArrayVec;
use derive_more::with_trait::{Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{num::Saturating, ops::Range, pin::Pin};

#[cfg(test)]
use proptest::prelude::*;

type MovesBuf<T = ()> = ArrayVec<(Move, T), 255>;

/// Indicates the search was interrupted .
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Error)]
#[display("the search was interrupted")]
pub struct Interrupt;

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
        moves: &[(Move, Value)],
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

            self.searcher.history.update(pos, best, draft.get());

            let counter = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
            counter.update(pos, best, draft.get());

            for &(m, _) in moves.iter().rev() {
                if m == best {
                    break;
                } else {
                    self.searcher.history.update(pos, m, -draft.get());

                    let counter = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
                    counter.update(pos, m, -draft.get());
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

        (idx >= 2 && self.value[idx] > self.value[idx - 2]) as i32
            + (idx >= 4 && self.value[idx] > self.value[idx - 4]) as i32
    }

    /// The mate distance pruning.
    fn mdp(&self, ply: Ply, bounds: &Range<Score>) -> (Score, Score) {
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    /// Computes the null move pruning reduction.
    fn nmp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        match surplus.get() {
            ..0 => None,
            s @ 0..20 => Some(draft - (s + 10) / 10 - draft / 4),
            20.. => Some(draft - 3 - draft / 4),
        }
    }

    /// Computes the multi-cut pruning reduction.
    fn mcp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        match draft.get() {
            ..6 => None,
            6.. => match surplus.get() {
                ..0 => None,
                0.. => Some(draft / 2),
            },
        }
    }

    /// Computes fail-high pruning reduction.
    fn fhp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        match surplus.get() {
            ..0 => None,
            s @ 0..360 => Some(draft - (s + 60) / 140),
            360.. => Some(draft - 3),
        }
    }

    /// Computes the fail-low pruning reduction.
    fn flp(&self, deficit: Score, draft: Depth) -> Option<Depth> {
        match deficit.get() {
            ..0 => None,
            s @ 0..900 => Some(draft - (s + 180) / 360),
            900.. => Some(draft - 3),
        }
    }

    /// Computes the late move reduction.
    fn lmr(&self, draft: Depth, idx: usize) -> Depth {
        (draft.get().max(1).ilog2() as i16 * idx.max(1).ilog2() as i16 / 2).saturate()
    }

    /// Computes the futility margin.
    fn futility(&self, draft: Depth) -> Value {
        (draft.cast::<i16>() * 80 + 60).saturate()
    }

    /// The zero-window alpha-beta search.
    fn nw<const N: usize>(
        &mut self,
        pos: &Evaluator,
        beta: Score,
        depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Interrupt> {
        self.ab(pos, beta - 1..beta, depth, ply)
    }

    /// The alpha-beta search.
    fn ab<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Interrupt> {
        if ply.cast::<usize>() < N && depth > ply && bounds.start + 1 < bounds.end {
            self.pvs(pos, bounds, depth, ply)
        } else {
            Ok(self.pvs::<0>(pos, bounds, depth, ply)?.truncate())
        }
    }

    /// The principal variation search.
    fn pvs<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        mut depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Interrupt> {
        (ply > 0).assume();
        self.nodes.update(1);
        if self.ctrl.check(self.root, &self.pv, ply) == ControlFlow::Abort {
            return Err(Interrupt);
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
                    if -self.nw::<0>(&next, -beta + 1, d + ply, ply + 1)? >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let killer = self.killers[ply.cast::<usize>()];
        let mut moves: MovesBuf<_> = pos
            .moves()
            .filter(|ms| !quiesce || !ms.is_quiet())
            .flatten()
            .map(|m| {
                if Some(m) == transposed.head() {
                    return (m, Value::upper());
                }

                let history = &self.searcher.history;
                let counter = self.replies[ply.cast::<usize>() - 1];
                let mut rating = pos.gain(m) + history.get(pos, m) + counter.get(pos, m);

                if killer.contains(m) {
                    rating += 128;
                }

                (m, rating)
            })
            .collect();

        moves.sort_unstable_by_key(|(_, rating)| *rating);
        let (mut head, mut tail) = match moves.last() {
            None => return Ok(transposed.truncate()),
            Some(&(m, _)) => {
                let mut sme = 0i8;
                if let Some(t) = transposition {
                    if let Some(d) = self.mcp(t.score().lower(ply) - beta, draft) {
                        if t.draft() >= d {
                            sme += 1;
                            for (m, _) in moves.iter().rev().skip(1) {
                                let mut next = pos.clone();
                                next.play(*m);
                                self.tt.prefetch(next.zobrist());
                                self.replies[ply.cast::<usize>()] =
                                    Some(self.searcher.continuation.reply(pos, *m));
                                if -self.nw::<0>(&next, -beta + 1, d + ply, ply + 1)? >= beta {
                                    return Ok(transposed.truncate());
                                }
                            }
                        }
                    }
                }

                let mut next = pos.clone();
                next.play(m);
                self.tt.prefetch(next.zobrist());
                self.replies[ply.cast::<usize>()] = Some(self.searcher.continuation.reply(pos, m));
                (m, -self.ab(&next, -beta..-alpha, depth + sme, ply + 1)?)
            }
        };

        let improving = self.improving(ply);
        for (idx, &(m, _)) in moves.iter().rev().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if idx as i32 > 1 + draft.cast::<i32>().pow(2) * (1 + improving) / 2 {
                break;
            }

            let lmr = self.lmr(draft, idx) - (is_pv as i8) - improving;
            let margin = alpha - self.value[ply.cast::<usize>()] - self.futility(draft - lmr);
            if !pos.winning(m, margin.saturate()) {
                continue;
            }

            let mut next = pos.clone();
            next.play(m);
            self.tt.prefetch(next.zobrist());
            self.replies[ply.cast::<usize>()] = Some(self.searcher.continuation.reply(pos, m));
            let pv = match -self.nw(&next, -alpha, depth - lmr, ply + 1)? {
                pv if pv <= alpha || (pv >= beta && lmr <= 0) => pv,
                _ => -self.ab(&next, -beta..-alpha, depth, ply + 1)?,
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
        moves: &mut [(Move, Value)],
        bounds: Range<Score>,
        depth: Depth,
    ) -> Result<Pv, Interrupt> {
        let ply = Ply::new(0);
        let (alpha, beta) = (bounds.start, bounds.end);
        if self.ctrl.check(self.root, &self.pv, ply) != ControlFlow::Continue {
            return Err(Interrupt);
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
        let mut tail = -self.ab(&next, -beta..-alpha, depth, ply + 1)?;

        for (idx, &(m, _)) in moves.iter().rev().skip(1).enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if idx as i32 > 1 + depth.cast::<i32>().pow(2) / 2 {
                break;
            }

            let lmr = self.lmr(depth, idx) - 1;
            let margin = alpha - self.value[0] - self.futility(depth - lmr);
            if !self.root.winning(m, margin.saturate()) {
                continue;
            }

            let mut next = self.root.clone();
            next.play(m);
            self.tt.prefetch(next.zobrist());
            self.nodes = Some(self.ctrl.attention().nodes(m));
            self.replies[0] = Some(self.searcher.continuation.reply(self.root, m));
            let pv = match -self.nw(&next, -alpha, depth - lmr, ply + 1)? {
                pv if pv <= alpha || (pv >= beta && lmr <= 0) => pv,
                _ => -self.ab(&next, -beta..-alpha, depth, ply + 1)?,
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
            let mut moves: MovesBuf<_> = pos.moves().flatten().map(|m| (m, pos.gain(m))).collect();
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

            loop {
                let pv = self.pv.clone().truncate();
                yield Info::new(depth, self.ctrl.time(), self.ctrl.nodes(), pv);
                if stop || depth >= Depth::upper() {
                    return;
                }

                depth += 1;
                let mut draft = depth;
                let mut delta = Saturating(5i16);
                let (mut lower, mut upper) = match depth.get() {
                    ..=4 => (Score::lower(), Score::upper()),
                    _ => (self.pv.score() - delta, self.pv.score() + delta),
                };

                loop {
                    let partial = match self.root(&mut moves, lower..upper, draft) {
                        Err(_) => break stop = true,
                        Ok(pv) => pv,
                    };

                    delta *= 2;
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
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s >= #b)] s: Score,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s < #b)] s: Score,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none())] s: Score,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn ab_returns_static_evaluation_if_max_ply(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        d: Depth,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);

        assert_eq!(
            stack.ab::<1>(&pos, Score::lower()..Score::upper(), d, Ply::upper()),
            Ok(Pv::empty(pos.evaluate().saturate()))
        );
    }

    #[proptest]
    fn ab_aborts_if_maximum_number_of_nodes_visited(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::Nodes(0));
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.ab::<1>(&pos, b, d, p), Err(Interrupt));
    }

    #[proptest]
    fn ab_aborts_if_time_is_up(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        thread::sleep(Duration::from_millis(1));
        assert_eq!(stack.ab::<1>(&pos, b, d, p), Err(Interrupt));
    }

    #[proptest]
    fn ab_can_be_aborted_upon_request(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        ctrl.abort();
        assert_eq!(stack.ab::<1>(&pos, b, d, p), Err(Interrupt));
    }

    #[proptest]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        #[by_ref] e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.ab::<1>(&pos, b, d, p), Ok(Pv::empty(Score::new(0))));
    }

    #[proptest]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut stack = Stack::new(&e.searchers[0], &e.tt, &ctrl, &pos);
        stack.pv = stack.pv.transpose(m);

        assert_eq!(stack.ab::<1>(&pos, b, d, p), Ok(Pv::empty(Score::mated(p))));
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
