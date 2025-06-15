use crate::chess::{Move, Position};
use crate::nnue::{Evaluator, Value};
use crate::syzygy::{Syzygy, Wdl};
use crate::util::{Assume, Bounded, Integer, Memory};
use crate::{params::Params, search::*};
use derive_more::with_trait::{Deref, DerefMut, Display, Error};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use futures::stream::{FusedStream, Stream, StreamExt};
use std::task::{Context, Poll};
use std::{ops::Range, pin::Pin, time::Duration};

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
    searcher: &'a Searcher,
    syzygy: &'a Syzygy,
    tt: &'a Memory<Transposition>,
    ctrl: &'a Control,
    nodes: Option<&'a Counter>,
    replies: [Option<&'a Reply>; Ply::MAX as usize + 1],
    killers: [Killers; Ply::MAX as usize + 1],
    value: [Value; Ply::MAX as usize + 1],
    evaluator: Evaluator,
    pv: Pv,
}

impl<'a> Stack<'a> {
    fn new(
        searcher: &'a Searcher,
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
            replies: [Default::default(); Ply::MAX as usize + 1],
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
        best: Move,
        score: Score,
        moves: &Moves,
    ) {
        let ply = self.evaluator.ply();
        let pos = &self.evaluator[ply];

        if score >= bounds.end {
            if best.is_quiet() {
                self.killers[ply.cast::<usize>()].insert(best);
            }

            let bonus_gamma = Params::history_bonus_gamma();
            let bonus_delta = Params::history_bonus_delta();
            let bonus = (depth.cast::<i32>() * bonus_gamma + bonus_delta) / Params::BASE;

            let penalty_gamma = Params::history_penalty_gamma();
            let penalty_delta = Params::history_penalty_delta();
            let penalty = -(depth.cast::<i32>() * penalty_gamma + penalty_delta) / Params::BASE;

            self.searcher.history.update(pos, best, bonus.saturate());

            let reply = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
            reply.update(pos, best, bonus.saturate());

            for m in moves.iter() {
                if m == best {
                    break;
                } else {
                    self.searcher.history.update(pos, m, penalty.saturate());

                    let reply = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
                    reply.update(pos, m, penalty.saturate());
                }
            }
        }

        let score = ScoreBound::new(bounds, score, ply);
        let tpos = Transposition::new(score, depth, Some(best));
        self.tt.set(pos.zobrist(), tpos);
    }

    /// A measure for how much the position is improving.
    fn improving(&self) -> i32 {
        if self.evaluator.is_check() {
            return 0;
        }

        let idx = self.evaluator.ply().cast::<usize>();

        let a = (idx >= 2 && self.value[idx] > self.value[idx - 2]) as i32;
        let b = (idx >= 4 && self.value[idx] > self.value[idx - 4]) as i32;

        a + b
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
        (gamma * depth.cast::<i32>() + delta) / Params::BASE
    }

    /// Computes the futility pruning threshold.
    fn fpt(&self, depth: Depth) -> i32 {
        let gamma = Params::futility_pruning_threshold_gamma();
        gamma * depth.cast::<i32>() / Params::BASE
    }

    /// Computes the SEE pruning threshold.
    fn spt(&self, depth: Depth) -> i32 {
        let gamma = Params::see_pruning_threshold_gamma();
        gamma * depth.cast::<i32>() / Params::BASE
    }

    /// Computes the late move reduction.
    fn lmr(&self, depth: Depth, idx: usize) -> i32 {
        let gamma = Params::late_move_reduction_gamma();
        let delta = Params::late_move_reduction_delta();

        let x = idx.max(1).ilog2() as i32;
        let y = depth.get().max(1).ilog2() as i32;
        (gamma * x * y + delta) / Params::BASE
    }

    /// Computes the late move pruning threshold.
    fn lmp(&self, depth: Depth, idx: usize) -> i32 {
        let gamma = Params::late_move_pruning_gamma();
        let delta = Params::late_move_pruning_delta();
        Params::BASE * idx.cast::<i32>() / (delta + gamma * depth.cast::<i32>().pow(2))
    }

    #[must_use]
    fn next(&mut self, m: Option<Move>) -> StackGuard<'_, 'a> {
        self.replies[self.evaluator.ply().cast::<usize>()] =
            m.map(|m| self.searcher.continuation.reply(&self.evaluator, m));

        self.evaluator.push(m);
        self.tt.prefetch(self.evaluator.zobrist());

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
        if self.ctrl.check(&self.evaluator, &self.pv) == ControlFlow::Abort {
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

        self.value[ply.cast::<usize>()] = self.evaluator.evaluate();
        let transposition = self.tt.get(self.evaluator.zobrist());
        let transposed = match transposition {
            None => Pv::empty(self.value[ply.cast::<usize>()].saturate()),
            Some(t) => t.transpose(ply),
        };

        if depth > 0 {
            depth += self.evaluator.is_check() as i8;
            depth -= transposition.is_none() as i8;
        }

        let is_pv = alpha + 1 < beta;
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
                        let transposition = Transposition::new(score, depth, None);
                        self.tt.set(self.evaluator.zobrist(), transposition);
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

        let killer_bonus = Params::killer_move_bonus();
        let gain_gamma = Params::noisy_gain_rating_gamma();
        let gain_delta = Params::noisy_gain_rating_delta();
        let killer = self.killers[ply.cast::<usize>()];

        moves.sort(|m| {
            if Some(m) == transposed.head() {
                return Bounded::upper();
            }

            let mut rating = Bounded::new(0);
            rating += self.searcher.history.get(&self.evaluator, m).cast::<i32>();

            let reply = self.replies.get(ply.cast::<usize>().wrapping_sub(1));
            rating += reply.get(&self.evaluator, m);

            if killer.contains(m) {
                rating += killer_bonus / Params::BASE;
            } else if !m.is_quiet() {
                let gain = self.evaluator.gain(m);
                if self.evaluator.winning(m, Value::new(1)) {
                    rating += (gain.cast::<i32>() * gain_gamma + gain_delta) / Params::BASE;
                }
            }

            rating
        });

        let mut extension = 0i8;
        if let Some(t) = transposition {
            if t.score().lower(ply) >= beta && t.depth() >= depth - 3 && depth >= 6 {
                extension = 2;
                let s_depth = (depth - 1) / 2;
                let s_beta = beta - self.single(depth);
                let d_beta = beta - self.double(depth);
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
                    }
                }
            }
        }

        let mut sorted_moves = moves.sorted();
        let (mut head, mut tail) = match sorted_moves.next() {
            None => return Ok(transposed.truncate()),
            Some(m) => {
                let mut next = self.next(Some(m));
                (m, -next.ab(depth + extension - 1, -beta..-alpha, false)?)
            }
        };

        let improving = self.improving();
        for (idx, m) in sorted_moves.enumerate() {
            let alpha = match tail.score() {
                s if s >= beta => break,
                s => s.max(alpha),
            };

            if self.lmp(depth, idx) > improving {
                break;
            } else if !self.evaluator.winning(m, Value::new(1) - self.spt(depth)) {
                continue;
            }

            let lmr = Depth::new(cut as _) + self.lmr(depth, idx) - is_pv as i8 - improving;
            if self.value[ply.cast::<usize>()] + self.futility(depth - lmr) <= alpha {
                let margin = Value::new(1) + self.fpt(depth - lmr);
                if !self.evaluator.winning(m, margin) {
                    continue;
                }
            }

            let mut next = self.next(Some(m));
            let pv = match -next.nw(depth - lmr - 1, -alpha, !cut || lmr > 0)? {
                pv if pv <= alpha || (pv >= beta && lmr <= 0) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        let tail = tail.clamp(lower, upper);
        self.record(depth, bounds, head, tail.score(), &moves);
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
        if self.ctrl.check(&self.evaluator, &self.pv) != ControlFlow::Continue {
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

            if self.lmp(depth, idx) > 0 {
                break;
            }

            let lmr = Depth::new(0) + self.lmr(depth, idx);
            if self.value[0] + self.futility(depth - lmr) <= alpha {
                let margin = Value::new(1) + self.fpt(depth - lmr);
                if !self.evaluator.winning(m, margin) {
                    continue;
                }
            }

            let mut next = self.next(Some(m));
            next.nodes = Some(next.ctrl.attention().nodes(m));
            let pv = match -next.nw(depth - lmr - 1, -alpha, false)? {
                pv if pv <= alpha || (pv >= beta && lmr <= 0) => pv,
                _ => -next.ab(depth - 1, -beta..-alpha, false)?,
            };

            if pv > tail {
                (head, tail) = (m, pv);
            }
        }

        self.record(depth, bounds, head, tail.score(), moves);
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
            if let Some(t) = self.tt.get(self.evaluator.zobrist()) {
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
        let searchers: &'static [Searcher] = unsafe { &*(&*self.engine.searchers as *const _) };
        let syzygy: &'static Syzygy = unsafe { &*(&self.engine.syzygy as *const _) };
        let tt: &'static Memory<Transposition> = unsafe { &*(&self.engine.tt as *const _) };
        let ctrl: &'static Control = unsafe { &*(&self.ctrl as *const _) };
        let position: &'static Position = unsafe { &*(self.position as *const _) };

        let (tx, rx) = unbounded();
        self.channel = Some(rx);
        if let Some(pv) = syzygy.best(position) {
            let info = Info::new(Depth::new(0), Duration::ZERO, 0, pv.truncate());
            return Poll::Ready(Some(info));
        }

        self.task = Some(executor.execute(move |idx| {
            let searcher = searchers.get(idx).assume();
            let evaluator = Evaluator::new(position.clone());
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
struct Searcher {
    history: History,
    continuation: Continuation,
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
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter((Value::lower()..Value::upper()).contains(&#b))] b: Score,
        d: Depth,
        #[filter(#s.mate() == Mate::None && #s >= #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Lower(s), Depth::upper(), Some(m));
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
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
        #[filter(#s.mate() == Mate::None && #s < #b)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Upper(s), Depth::upper(), Some(m));
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
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
        #[filter(#s.mate() == Mate::None)] s: Score,
        cut: bool,
    ) {
        let tpos = Transposition::new(ScoreBound::Exact(s), Depth::upper(), Some(m));
        e.tt.set(pos.zobrist(), tpos);

        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.nw::<1>(d, b, cut), Ok(Pv::empty(s)));
    }

    #[proptest]
    fn ab_aborts_if_maximum_number_of_nodes_visited(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::nodes(0));
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.pv = stack.pv.transpose(m);
        assert_eq!(stack.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_aborts_if_time_is_up(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::time(Duration::ZERO));
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.pv = stack.pv.transpose(m);
        thread::sleep(Duration::from_millis(1));
        assert_eq!(stack.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_can_be_aborted_upon_request(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Evaluator,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        stack.pv = stack.pv.transpose(m);
        ctrl.abort();
        assert_eq!(stack.ab::<1>(d, b, cut), Err(Interrupted));
    }

    #[proptest]
    fn ab_returns_drawn_score_if_game_ends_in_a_draw(
        e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_draw()))] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        assert_eq!(stack.ab::<1>(d, b, cut), Ok(Pv::empty(Score::new(0))));
    }

    #[proptest]
    fn ab_returns_lost_score_if_game_ends_in_checkmate(
        e: Engine,
        #[filter(#pos.outcome().is_some_and(|o| o.is_decisive()))] pos: Evaluator,
        #[filter(!#b.is_empty())] b: Range<Score>,
        d: Depth,
        cut: bool,
    ) {
        let ply = pos.ply();
        let ctrl = Control::new(&pos, Limits::none());
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, pos);
        assert_eq!(stack.ab::<1>(d, b, cut), Ok(Pv::empty(Score::mated(ply))));
    }

    #[proptest]
    fn aw_extends_time_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let evaluator = Evaluator::new(pos);
        let ctrl = Control::new(&evaluator, Limits::time(Duration::ZERO));
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, evaluator);
        let last = stack.aw().last();
        assert_ne!(last.and_then(|pv| pv.head()), None);
    }

    #[proptest]
    fn aw_extends_depth_to_find_some_pv(
        e: Engine,
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        let evaluator = Evaluator::new(pos);
        let ctrl = Control::new(&evaluator, Limits::depth(Depth::lower()));
        let mut stack = Stack::new(&e.searchers[0], &e.syzygy, &e.tt, &ctrl, evaluator);
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
