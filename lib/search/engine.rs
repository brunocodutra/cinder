use crate::chess::{Move, Position};
use crate::nnue::{Evaluator, Value};
use crate::search::*;
use crate::util::{Assume, Integer};
use arrayvec::ArrayVec;
use derive_more::with_trait::{Constructor, Deref, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::{Stream, StreamExt, stream::FusedStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::{num::Saturating, ops::Range, pin::Pin, thread, time::Duration};

#[cfg(test)]
use proptest::strategy::LazyJust;

type MovesBuf<T = ()> = ArrayVec<(Move, T), 255>;

/// Indicates the search was aborted .
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Error)]
#[display("the search was aborted")]
pub struct Aborted;

/// Information about the search result.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Info<const N: usize = { Depth::MAX as _ }> {
    depth: Depth,
    time: Duration,
    nodes: u64,
    #[deref]
    pv: Pv<N>,
}

impl<const N: usize> Info<N> {
    /// The depth searched.
    pub fn depth(&self) -> Depth {
        self.depth
    }

    /// The duration searched.
    pub fn time(&self) -> Duration {
        self.time
    }

    /// The number of nodes searched.
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// The number of nodes searched per second.
    pub fn nps(&self) -> f64 {
        self.nodes as f64 / self.time().as_secs_f64()
    }

    /// The principal variation.
    pub fn pv(&self) -> &Pv<N> {
        &self.pv
    }
}

#[derive(Debug, Clone)]
struct Worker<'a> {
    ctrl: &'a Control,
    tt: &'a TranspositionTable,
    history: &'a History,
    continuation: &'a Continuation,
    replies: [Option<&'a Reply>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
}

impl<'a> Worker<'a> {
    fn new(
        ctrl: &'a Control,
        tt: &'a TranspositionTable,
        history: &'a History,
        continuation: &'a Continuation,
    ) -> Self {
        Worker {
            ctrl,
            tt,
            history,
            continuation,
            replies: [Default::default(); Ply::MAX as usize + 1],
            killers: [Default::default(); Ply::MAX as usize + 1],
            value: [Default::default(); Ply::MAX as usize + 1],
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

            self.history.update(pos, best, draft.get());

            let counter = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
            counter.update(pos, best, draft.get());

            for &(m, _) in moves.iter().rev() {
                if m == best {
                    break;
                } else {
                    self.history.update(pos, m, -draft.get());

                    let counter = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
                    counter.update(pos, m, -draft.get())
                }
            }
        }

        let score = ScoreBound::new(bounds, score, ply);
        let tpos = Transposition::new(score, draft, best);
        self.tt.set(pos.zobrist(), tpos);
    }

    fn info<const N: usize>(&self, depth: Depth, pv: Pv<N>) -> Info<N> {
        Info::new(depth, self.ctrl.time(), self.ctrl.nodes(), pv)
    }

    /// An implementation of the [improving heuristic].
    ///
    /// [improving heuristic]: https://www.chessprogramming.org/Improving
    fn improving(&mut self, ply: Ply) -> i32 {
        let idx = ply.cast::<usize>();

        (idx >= 2 && self.value[idx] > self.value[idx - 2]) as i32
            + (idx >= 4 && self.value[idx] > self.value[idx - 4]) as i32
    }

    /// An implementation of [mate distance pruning].
    ///
    /// [mate distance pruning]: https://www.chessprogramming.org/Mate_Distance_Pruning
    fn mdp(&self, ply: Ply, bounds: &Range<Score>) -> (Score, Score) {
        let lower = Score::mated(ply);
        let upper = Score::mating(ply + 1); // One can't mate in 0 plies!
        (bounds.start.max(lower), bounds.end.min(upper))
    }

    /// An implementation of [null move pruning].
    ///
    /// [null move pruning]: https://www.chessprogramming.org/Null_Move_Pruning
    fn nmp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        match surplus.get() {
            ..0 => None,
            s @ 0..20 => Some(draft - (s + 10) / 10 - draft / 4),
            20.. => Some(draft - 3 - draft / 4),
        }
    }

    /// An implementation of [multi-cut pruning].
    ///
    /// [multi-cut pruning]: https://www.chessprogramming.org/Multi-Cut
    fn mcp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        match draft.get() {
            ..6 => None,
            6.. => match surplus.get() {
                ..0 => None,
                0.. => Some(draft / 2),
            },
        }
    }

    /// An implementation of [reverse futility pruning].
    ///
    /// [reverse futility pruning]: https://www.chessprogramming.org/Reverse_Futility_Pruning
    fn rfp(&self, surplus: Score, draft: Depth) -> Option<Depth> {
        match surplus.get() {
            ..0 => None,
            s @ 0..360 => Some(draft - (s + 60) / 140),
            360.. => Some(draft - 3),
        }
    }

    /// An implementation of [late move reductions].
    ///
    /// [late move reductions]: https://www.chessprogramming.org/Late_Move_Reductions
    fn lmr(&self, draft: Depth, idx: usize) -> Depth {
        (draft.get().max(1).ilog2() as i16 * idx.max(1).ilog2() as i16 / 2).saturate()
    }

    /// The [zero-window] alpha-beta search.
    ///
    /// [zero-window]: https://www.chessprogramming.org/Null_Window
    fn nw<const N: usize>(
        &mut self,
        pos: &Evaluator,
        beta: Score,
        depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Aborted> {
        self.ab(pos, beta - 1..beta, depth, ply)
    }

    /// The [alpha-beta] search.
    ///
    /// [alpha-beta]: https://www.chessprogramming.org/Alpha-Beta
    fn ab<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Aborted> {
        if ply.cast::<usize>() < N && depth > ply && bounds.start + 1 < bounds.end {
            self.recurse(pos, bounds, depth, ply)
        } else {
            Ok(self.recurse::<0>(pos, bounds, depth, ply)?.truncate())
        }
    }

    fn recurse<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        mut depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Aborted> {
        if self.ctrl.check() == ControlFlow::Abort {
            return Err(Aborted);
        }

        (ply > 0).assume();
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

            if let Some(d) = self.rfp(lower - beta, draft) {
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

                let counter = self.replies[ply.cast::<usize>() - 1];
                let mut rating = pos.gain(m) + self.history.get(pos, m) + counter.get(pos, m);

                if killer.contains(m) {
                    rating += 128;
                }

                (m, rating)
            })
            .collect();

        moves.sort_unstable_by_key(|(_, rating)| *rating);

        if let Some(t) = transposition {
            if let Some(d) = self.mcp(t.score().lower(ply) - beta, draft) {
                if t.draft() >= d {
                    depth += 1;
                    for (m, _) in moves.iter().rev().skip(1) {
                        let mut next = pos.clone();
                        next.play(*m);
                        self.tt.prefetch(next.zobrist());
                        self.replies[ply.cast::<usize>()] = Some(self.continuation.reply(pos, *m));
                        if -self.nw::<0>(&next, -beta + 1, d + ply, ply + 1)? >= beta {
                            return Ok(transposed.truncate());
                        }
                    }
                }
            }
        }

        match self.pvs(pos, &moves, alpha..beta, depth, ply)? {
            None => Ok(transposed.truncate()),
            Some(pv) => Ok(pv),
        }
    }

    /// An implementation of [PVS].
    ///
    /// [PVS]: https://www.chessprogramming.org/Principal_Variation_Search
    fn pvs<const N: usize>(
        &mut self,
        pos: &Evaluator,
        moves: &[(Move, Value)],
        bounds: Range<Score>,
        depth: Depth,
        ply: Ply,
    ) -> Result<Option<Pv<N>>, Aborted> {
        let (alpha, beta) = (bounds.start, bounds.end);
        let is_pv = alpha + 1 < beta;
        let draft = depth - ply;

        let (mut head, mut tail) = match moves.last() {
            None => return Ok(None),
            Some(&(m, _)) => {
                let mut next = pos.clone();
                next.play(m);
                self.tt.prefetch(next.zobrist());
                self.replies[ply.cast::<usize>()] = Some(self.continuation.reply(pos, m));
                (m, -self.ab(&next, -beta..-alpha, depth, ply + 1)?)
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

            let mut next = pos.clone();
            next.play(m);

            self.tt.prefetch(next.zobrist());
            let lmr = self.lmr(draft, idx) - (is_pv as i8) - improving;
            self.replies[ply.cast::<usize>()] = Some(self.continuation.reply(pos, m));
            let partial = match -self.nw(&next, -alpha, depth - lmr, ply + 1)? {
                partial if partial <= alpha || (partial >= beta && lmr <= 0) => partial,
                _ => -self.ab(&next, -beta..-alpha, depth, ply + 1)?,
            };

            if partial > tail {
                (head, tail) = (m, partial);
            }
        }

        self.record(pos, moves, bounds, depth, ply, head, tail.score());
        Ok(Some(tail.transpose(head)))
    }

    /// An implementation of [aspiration windows] with [iterative deepening].
    ///
    /// [aspiration windows]: https://www.chessprogramming.org/Aspiration_Windows
    /// [iterative deepening]: https://www.chessprogramming.org/Iterative_Deepening
    fn aw<const N: usize>(&mut self, pos: &Evaluator) -> impl IntoIterator<Item = Info<N>> {
        gen move {
            self.value[0] = pos.evaluate();
            let mut depth = Depth::new(0);
            let mut moves: MovesBuf<_> = pos.moves().flatten().map(|m| (m, pos.gain(m))).collect();
            let mut pv = match moves.iter().max_by_key(|(_, rating)| *rating) {
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
                yield self.info(depth, pv.clone());
                if stop || depth >= limits.depth() {
                    return;
                }

                depth += 1;
                let mut draft = depth;
                let mut delta = Saturating(5i16);
                let (mut lower, mut upper) = match depth.get() {
                    ..=4 => (Score::lower(), Score::upper()),
                    _ => (pv.score() - delta, pv.score() + delta),
                };

                loop {
                    if self.ctrl.check() != ControlFlow::Continue {
                        break stop = true;
                    }

                    for (m, rating) in moves.iter_mut() {
                        if Some(*m) == pv.head() {
                            *rating = Value::upper();
                        } else {
                            *rating = pos.gain(*m) + self.history.get(pos, *m);
                        }
                    }

                    moves.sort_unstable_by_key(|(_, rating)| *rating);
                    let partial = match self.pvs(pos, &moves, lower..upper, draft, Ply::new(0)) {
                        Ok(partial) => partial.assume(),
                        Err(_) => break stop = true,
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
                            pv = partial;
                        }

                        _ => break pv = partial,
                    }
                }
            }
        }
    }
}

/// A handle to an ongoing search.
#[derive(Debug)]
pub struct Search<'e, 'p, 'c> {
    engine: &'e mut Engine,
    position: &'p Evaluator,
    ctrl: &'c Control,
    workers: AtomicUsize,
    channel: Option<UnboundedReceiver<Info>>,
}

impl<'e, 'p, 'c> Search<'e, 'p, 'c> {
    fn new(engine: &'e mut Engine, position: &'p Evaluator, ctrl: &'c Control) -> Self {
        Search {
            engine,
            position,
            ctrl,
            workers: AtomicUsize::new(0),
            channel: None,
        }
    }
}

impl<'e, 'p, 'c> Drop for Search<'e, 'p, 'c> {
    fn drop(&mut self) {
        self.ctrl.abort();
        while self.workers.load(Ordering::Acquire) > 0 {
            thread::yield_now();
        }
    }
}

impl<'e, 'p, 'c> FusedStream for Search<'e, 'p, 'c> {
    fn is_terminated(&self) -> bool {
        self.channel.as_ref().is_some_and(|c| c.is_terminated())
    }
}

impl<'e, 'p, 'c> Stream for Search<'e, 'p, 'c> {
    type Item = Info;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(channel) = &mut self.channel {
            return channel.poll_next_unpin(cx);
        }

        let (tx, mut rx) = unbounded();

        *self.workers.get_mut() = self.engine.threads.get();
        let pos: &'static _ = unsafe { &*(self.position as *const _) };
        let ctrl: &'static _ = unsafe { &*(self.ctrl as *const _) };
        let tt: &'static _ = unsafe { &*(&self.engine.tt as *const _) };
        let history: &'static _ = unsafe { &*(&self.engine.history as *const _) };
        let continuation: &'static _ = unsafe { &*(&self.engine.continuation as *const _) };
        let workers: &'static AtomicUsize = unsafe { &*(&self.workers as *const _) };

        thread::spawn(move || {
            for info in Worker::new(ctrl, tt, history, continuation).aw(pos) {
                tx.unbounded_send(info).assume();
            }

            drop(tx);
            workers.fetch_sub(1, Ordering::Release);
        });

        for _ in 1..self.engine.threads.get() {
            thread::spawn(|| {
                for _ in Worker::new(ctrl, tt, history, continuation).aw::<1>(pos) {}
                workers.fetch_sub(1, Ordering::Release);
            });
        }

        let poll = rx.poll_next_unpin(cx);
        self.channel = Some(rx);
        poll
    }
}

/// A chess engine.
#[derive(Debug)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Engine {
    threads: ThreadCount,
    #[cfg_attr(test, map(|s: HashSize| TranspositionTable::new(s)))]
    tt: TranspositionTable,
    #[cfg_attr(test, strategy(LazyJust::new(History::default)))]
    history: History,
    #[cfg_attr(test, strategy(LazyJust::new(Continuation::default)))]
    continuation: Continuation,
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
            threads: options.threads,
            tt: TranspositionTable::new(options.hash),
            history: History::default(),
            continuation: Continuation::default(),
        }
    }

    /// Initiates a [`Search`].
    pub fn search<'e, 'p, 'c>(
        &'e mut self,
        pos: &'p Evaluator,
        ctrl: &'c Control,
    ) -> Search<'e, 'p, 'c> {
        Search::new(self, pos, ctrl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{prop_assume, sample::Selector};
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
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s >= #b)] s: Score,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        assert_eq!(worker.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s < #b)] s: Score,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        assert_eq!(worker.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
        #[filter(#s.mate().is_none())] s: Score,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), d, m);
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        assert_eq!(worker.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn ab_returns_static_evaluation_if_max_ply(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        d: Depth,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);

        assert_eq!(
            worker.ab::<1>(&pos, Score::lower()..Score::upper(), d, Ply::upper()),
            Ok(Pv::empty(pos.evaluate().saturate()))
        );
    }

    #[proptest]
    fn ab_aborts_if_maximum_number_of_nodes_visited(
        e: Engine,
        pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::Nodes(0));
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        assert_eq!(worker.ab::<1>(&pos, b, d, p), Err(Aborted));
    }

    #[proptest]
    fn ab_aborts_if_time_is_up(
        e: Engine,
        pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        thread::sleep(Duration::from_millis(1));
        assert_eq!(worker.ab::<1>(&pos, b, d, p), Err(Aborted));
    }

    #[proptest]
    fn ab_can_be_aborted_upon_request(
        e: Engine,
        pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        ctrl.abort();
        assert_eq!(worker.ab::<1>(&pos, b, d, p), Err(Aborted));
    }

    #[proptest]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        #[by_ref] e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        assert_eq!(worker.ab::<1>(&pos, b, d, p), Ok(Pv::empty(Score::new(0))));
    }

    #[proptest]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let ctrl = Control::new(&pos, Limits::None);
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);

        assert_eq!(
            worker.ab::<1>(&pos, b, d, p),
            Ok(Pv::empty(Score::mated(p)))
        );
    }

    #[proptest]
    fn aw_extends_time_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        let last = worker.aw::<1>(&pos).into_iter().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_depth_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let ctrl = Control::new(&pos, Limits::Depth(Depth::lower()));
        let mut worker = Worker::new(&ctrl, &e.tt, &e.history, &e.continuation);
        let last = worker.aw::<1>(&pos).into_iter().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }
}
