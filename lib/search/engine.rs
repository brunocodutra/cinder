use crate::chess::{Move, Outcome, Position};
use crate::nnue::{Evaluator, Value};
use crate::search::*;
use crate::util::{Assume, Counter, Integer, Timer, Trigger};
use arrayvec::ArrayVec;
use derive_more::Deref;
use std::{mem::swap, ops::Range, thread, time::Duration};

#[cfg(test)]
use proptest::strategy::LazyJust;

/// A chess engine.
#[derive(Debug, Clone, Deref)]
pub struct Search<'a> {
    #[deref]
    engine: &'a Engine,
    ctrl: Control<'a>,
    value: [Value; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    continuation: [Option<&'a Reply>; Ply::MAX as usize + 1],
}

impl<'a> Search<'a> {
    fn new(engine: &'a Engine, ctrl: Control<'a>) -> Self {
        let value = [Default::default(); Ply::MAX as usize + 1];
        let killers = [Default::default(); Ply::MAX as usize + 1];
        let continuation = [Default::default(); Ply::MAX as usize + 1];

        Search {
            engine,
            ctrl,
            value,
            killers,
            continuation,
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

            let counter = self.continuation.get(ply.cast::<usize>().wrapping_sub(1));
            counter.update(pos, best, draft.get());

            for &(m, _) in moves.iter().rev() {
                if m == best {
                    break;
                } else {
                    self.history.update(pos, m, -draft.get());

                    let counter = self.continuation.get(ply.cast::<usize>().wrapping_sub(1));
                    counter.update(pos, m, -draft.get())
                }
            }
        }

        let score = ScoreBound::new(bounds, score, ply);
        let tpos = Transposition::new(score, draft, best);
        self.tt.set(pos.zobrist(), tpos);
    }

    /// An implementation of the [improving heuristic].
    ///
    /// [improving heuristic]: https://www.chessprogramming.org/Improving
    fn improving(&mut self, ply: Ply) -> i8 {
        let idx = ply.cast::<usize>();

        (idx >= 2 && self.value[idx] > self.value[idx - 2]) as i8
            + (idx >= 4 && self.value[idx] > self.value[idx - 4]) as i8
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
            s @ 0..40 => Some(draft - (s + 20) / 20 - draft / 4),
            40.. => Some(draft - 3 - draft / 4),
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

    /// An implementation of [razoring].
    ///
    /// [razoring]: https://www.chessprogramming.org/Razoring
    fn razor(&self, deficit: Score, draft: Depth) -> Option<Depth> {
        match deficit.get() {
            ..0 => None,
            s @ 0..600 => Some(draft - (s + 30) / 210),
            600.. => Some(draft - 3),
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

    /// An implementation of [futility pruning].
    ///
    /// [futility pruning]: https://www.chessprogramming.org/Futility_Pruning
    fn fp(&self, deficit: Score, draft: Depth) -> Option<Depth> {
        match deficit.get() {
            ..0 => None,
            d @ 0..90 => Some(draft - (d + 30) / 40),
            90.. => Some(draft - 3),
        }
    }

    /// An implementation of [late move reductions].
    ///
    /// [late move reductions]: https://www.chessprogramming.org/Late_Move_Reductions
    fn lmr(&self, draft: Depth, idx: usize) -> Depth {
        (draft.get().max(1).ilog2() as i16 * idx.max(1).ilog2() as i16 / 2).saturate()
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
    ) -> Result<Pv<N>, Interrupted> {
        if ply.cast::<usize>() < N && depth > ply && bounds.start + 1 < bounds.end {
            self.pvs(pos, bounds, depth, ply)
        } else {
            Ok(self.pvs::<0>(pos, bounds, depth, ply)?.truncate())
        }
    }

    /// The full-window alpha-beta search.
    fn fw<const N: usize>(
        &mut self,
        pos: &Evaluator,
        depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Interrupted> {
        self.ab(pos, Score::lower()..Score::upper(), depth, ply)
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
    ) -> Result<Pv<N>, Interrupted> {
        self.ab(pos, beta - 1..beta, depth, ply)
    }

    /// An implementation of the [PVS] variation of the alpha-beta search.
    ///
    /// [PVS]: https://www.chessprogramming.org/Principal_Variation_Search
    fn pvs<const N: usize>(
        &mut self,
        pos: &Evaluator,
        bounds: Range<Score>,
        mut depth: Depth,
        ply: Ply,
    ) -> Result<Pv<N>, Interrupted> {
        self.ctrl.interrupted()?;

        let is_root = ply == 0;
        let (alpha, beta) = match pos.outcome() {
            None => self.mdp(ply, &bounds),
            Some(Outcome::DrawByThreefoldRepetition) if is_root => self.mdp(ply, &bounds),
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

        if !is_root && transposition.is_some() && pos.is_check() {
            depth = depth + 1;
        } else if !is_root && transposition.is_none() && !pos.is_check() {
            depth = depth - 2;
        }

        let draft = depth - ply;
        let quiesce = draft <= 0;
        let is_pv = alpha + 1 < beta;
        if let Some(t) = transposition {
            let (lower, upper) = t.score().range(ply).into_inner();

            #[allow(clippy::collapsible_if)]
            if lower >= upper || upper <= alpha || lower >= beta {
                if !is_pv && t.draft() >= draft {
                    return Ok(transposed.truncate());
                }
            }

            if let Some(d) = self.razor(alpha - upper, draft) {
                if !is_pv && t.draft() >= d {
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
                    self.continuation[ply.cast::<usize>()] = None;
                    if -self.nw::<0>(&next, -beta + 1, d + ply, ply + 1)? >= beta {
                        return Ok(transposed.truncate());
                    }
                }
            }
        }

        let killer = self.killers[ply.cast::<usize>()];
        let mut moves: ArrayVec<_, 255> = pos
            .moves()
            .filter(|ms| !quiesce || !ms.is_quiet())
            .flatten()
            .map(|m| {
                if Some(m) == transposed.head() {
                    return (m, Value::upper());
                } else if killer.contains(m) {
                    return (m, Value::new(128));
                }

                let gain = if m.is_quiet() {
                    Value::new(0)
                } else {
                    pos.gain(m)
                };

                let counter = self.continuation.get(ply.cast::<usize>().wrapping_sub(1));
                (m, gain + self.history.get(pos, m) + counter.get(pos, m))
            })
            .collect();

        moves.sort_unstable_by_key(|(_, gain)| *gain);

        if let Some(t) = transposition {
            if let Some(d) = self.mcp(t.score().lower(ply) - beta, draft) {
                if !is_root && t.draft() >= d {
                    depth = depth + 1;
                    for (m, _) in moves.iter().rev().skip(1) {
                        let mut next = pos.clone();
                        next.play(*m);
                        self.tt.prefetch(next.zobrist());
                        self.continuation[ply.cast::<usize>()] =
                            Some(self.engine.continuation.reply(pos, *m));
                        if -self.nw::<0>(&next, -beta + 1, d + ply, ply + 1)? >= beta {
                            return Ok(transposed.truncate());
                        }
                    }
                }
            }
        }

        let (mut head, mut tail) = match moves.pop() {
            None => return Ok(transposed.truncate()),
            Some((m, _)) => {
                let mut next = pos.clone();
                next.play(m);
                self.tt.prefetch(next.zobrist());
                self.continuation[ply.cast::<usize>()] =
                    Some(self.engine.continuation.reply(pos, m));
                (m, -self.ab(&next, -beta..-alpha, depth, ply + 1)?)
            }
        };

        if tail >= beta || moves.is_empty() {
            self.record(pos, &[], bounds, depth, ply, head, tail.score());
            return Ok(head >> tail);
        }

        let improving = self.improving(ply);
        for (idx, &(m, gain)) in moves.iter().rev().enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            let mut next = pos.clone();
            next.play(m);

            self.tt.prefetch(next.zobrist());
            if gain < 0 && draft < 4 && !pos.is_check() && !next.is_check() {
                let deficit = alpha + next.evaluate();
                if self.fp(deficit, draft).is_some_and(|d| d <= 0) {
                    break;
                }
            }

            let lmr = self.lmr(draft, idx) - (is_pv as i8) - improving;
            self.continuation[ply.cast::<usize>()] = Some(self.engine.continuation.reply(pos, m));
            let partial = match -self.nw(&next, -alpha, depth - lmr, ply + 1)? {
                partial if partial <= alpha || (partial >= beta && lmr <= 0) => partial,
                _ => -self.ab(&next, -beta..-alpha, depth, ply + 1)?,
            };

            if partial > tail {
                (head, tail) = (m, partial);
            }
        }

        self.record(pos, &moves, bounds, depth, ply, head, tail.score());
        Ok(head >> tail)
    }

    /// An implementation of [aspiration windows] with [iterative deepening].
    ///
    /// [aspiration windows]: https://www.chessprogramming.org/Aspiration_Windows
    /// [iterative deepening]: https://www.chessprogramming.org/Iterative_Deepening
    fn aw<const N: usize>(
        &mut self,
        pos: &Evaluator,
        limit: Depth,
        time: Range<Duration>,
    ) -> Pv<N> {
        let mut ctrl = Control::Unlimited;
        swap(&mut self.ctrl, &mut ctrl);
        self.fw::<0>(pos, Depth::new(0), Ply::new(0)).assume();
        let mut pv = self.fw(pos, Depth::new(1), Ply::new(0)).assume();
        swap(&mut self.ctrl, &mut ctrl);

        let mut depth = Depth::new(1);
        'id: while depth < limit {
            depth = depth + 1;
            let mut draft = depth;
            let mut delta = 5i16;

            let (mut lower, mut upper) = match depth.get() {
                ..=4 => (Score::lower(), Score::upper()),
                _ => (pv.score() - delta, pv.score() + delta),
            };

            'aw: loop {
                delta = delta.saturating_mul(2);
                if self.ctrl.timer().remaining() < Some(time.end - time.start) {
                    break 'id;
                }

                let Ok(partial) = self.ab(pos, lower..upper, draft, Ply::new(0)) else {
                    break 'id;
                };

                match partial.score() {
                    score if (-lower..Score::upper()).contains(&-score) => {
                        upper = lower / 2 + upper / 2;
                        lower = score - delta;
                        draft = depth;
                    }

                    score if (upper..Score::upper()).contains(&score) => {
                        upper = score + delta;
                        draft = draft - 1;
                        pv = partial;
                    }

                    _ => {
                        pv = partial;
                        break 'aw;
                    }
                }
            }
        }

        pv
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

    fn time_to_search(&self, pos: &Position, limits: &Limits) -> Range<Duration> {
        let (clock, inc) = match limits {
            Limits::Clock(c, i) => (c, i),
            _ => return limits.time()..limits.time(),
        };

        let time_left = clock.saturating_sub(*inc);
        let moves_left = 280 / pos.fullmoves().get().min(40);
        let time_per_move = inc.saturating_add(time_left / moves_left);
        time_per_move / 2..time_per_move
    }

    /// Searches for the [principal variation][`Pv`].
    pub fn search(&self, pos: &Evaluator, limits: &Limits, stopper: &Trigger) -> Pv {
        let time = self.time_to_search(pos, limits);
        let nodes = Counter::new(limits.nodes());
        let timer = Timer::new(time.end);
        let ctrl = Control::Limited(&nodes, &timer, stopper);
        let mut search = Search::new(self, ctrl);

        thread::scope(|s| {
            for _ in 1..self.threads.get() {
                let time = time.clone();
                let mut search = search.clone();
                s.spawn(move || search.aw::<0>(pos, limits.depth(), time));
            }

            let pv = search.aw(pos, limits.depth(), time);
            stopper.disarm();
            pv
        })
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
        #[filter(#p >= 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s >= #b)] s: Score,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), d, m);
        e.tt.set(pos.zobrist(), tpos);
        let mut search = Search::new(&e, Control::Unlimited);
        assert_eq!(search.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_beta_too_high(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p >= 0)] p: Ply,
        #[filter(#s.mate().is_none() && #s < #b)] s: Score,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), d, m);
        e.tt.set(pos.zobrist(), tpos);
        let mut search = Search::new(&e, Control::Unlimited);
        assert_eq!(search.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn nw_returns_transposition_if_exact(
        #[by_ref]
        #[filter(#e.tt.capacity() > 0)]
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#p >= 0)] p: Ply,
        #[filter(#s.mate().is_none())] s: Score,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), d, m);
        e.tt.set(pos.zobrist(), tpos);
        let mut search = Search::new(&e, Control::Unlimited);
        assert_eq!(search.nw::<1>(&pos, b, d, p), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn ab_aborts_if_maximum_number_of_nodes_visited(
        e: Engine,
        pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let nodes = Counter::new(0);
        let timer = Timer::infinite();
        let trigger = Trigger::armed();
        let ctrl = Control::Limited(&nodes, &timer, &trigger);
        let mut search = Search::new(&e, ctrl);
        assert_eq!(search.ab::<1>(&pos, b, d, p), Err(Interrupted));
    }

    #[proptest]
    fn ab_aborts_if_time_is_up(
        e: Engine,
        pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let nodes = Counter::new(u64::MAX);
        let timer = Timer::new(Duration::ZERO);
        let trigger = Trigger::armed();
        let ctrl = Control::Limited(&nodes, &timer, &trigger);
        let mut search = Search::new(&e, ctrl);
        std::thread::sleep(Duration::from_millis(1));
        assert_eq!(search.ab::<1>(&pos, b, d, p), Err(Interrupted));
    }

    #[proptest]
    fn ab_aborts_if_stopper_is_disarmed(
        e: Engine,
        pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let nodes = Counter::new(u64::MAX);
        let timer = Timer::infinite();
        let trigger = Trigger::disarmed();
        let ctrl = Control::Limited(&nodes, &timer, &trigger);
        let mut search = Search::new(&e, ctrl);
        assert_eq!(search.ab::<1>(&pos, b, d, p), Err(Interrupted));
    }

    #[proptest]
    fn ab_returns_static_evaluation_if_max_ply(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
    ) {
        let mut search = Search::new(&e, Control::Unlimited);

        assert_eq!(
            search.ab::<1>(&pos, b, d, Ply::upper()),
            Ok(Pv::empty(pos.evaluate().saturate()))
        );
    }

    #[proptest]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        #[by_ref] e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0 || #pos.outcome() != Some(Outcome::DrawByThreefoldRepetition))] p: Ply,
    ) {
        let mut search = Search::new(&e, Control::Unlimited);

        assert_eq!(search.ab::<1>(&pos, b, d, p), Ok(Pv::empty(Score::new(0))));
    }

    #[proptest]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        #[filter(#p > 0)] p: Ply,
    ) {
        let mut search = Search::new(&e, Control::Unlimited);

        assert_eq!(
            search.ab::<1>(&pos, b, d, p),
            Ok(Pv::empty(Score::mated(p)))
        );
    }

    #[proptest]
    fn search_extends_time_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let limits = Duration::ZERO.into();
        let trigger = Trigger::armed();
        assert_ne!(e.search(&pos, &limits, &trigger).head(), None);
    }

    #[proptest]
    fn search_extends_depth_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let limits = Depth::lower().into();
        let trigger = Trigger::armed();
        assert_ne!(e.search(&pos, &limits, &trigger).head(), None);
    }

    #[proptest]
    fn search_ignores_stopper_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
    ) {
        let limits = Limits::None;
        let trigger = Trigger::armed();
        assert_ne!(e.search(&pos, &limits, &trigger).head(), None);
    }
}
