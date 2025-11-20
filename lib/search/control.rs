use crate::search::{Attention, Limits, Ply, Pv};
use crate::{chess::Position, params::Params, util::Integer};
use derive_more::with_trait::Deref;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Controls the search flow.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ControlFlow {
    /// Continue searching.
    Continue,
    /// Interrupt searching as soon as possible.
    Stop,
    /// Interrupt searching immediately.
    Abort,
}

/// The global search control.
#[derive(Debug)]
pub struct GlobalControl {
    abort: AtomicBool,
    visited: AtomicU64,
    time: Range<f64>,
    timestamp: Instant,
    limits: Limits,
}

impl GlobalControl {
    #[inline(always)]
    fn time_to_search(pos: &Position, limits: &Limits) -> Range<f64> {
        let (clock, inc) = match limits.clock {
            Some((clock, inc)) => (clock.as_secs_f64(), inc.as_secs_f64()),
            None => return f64::INFINITY..limits.max_time().as_secs_f64(),
        };

        let time_left = clock - inc;
        let moves_left_start = Params::moves_left_start()[0] as f64 / Params::BASE as f64;
        let moves_left_end = Params::moves_left_end()[0] as f64 / Params::BASE as f64;
        let max_fullmoves = moves_left_start / moves_left_end;
        let moves_left = moves_left_start / max_fullmoves.min(pos.fullmoves().get() as _);
        let time_per_move = inc + time_left / moves_left;

        let soft_time_fraction = Params::soft_time_fraction()[0] as f64 / Params::BASE as f64;
        let hard_time_fraction = Params::hard_time_fraction()[0] as f64 / Params::BASE as f64;
        soft_time_fraction * time_per_move..clock * hard_time_fraction
    }

    /// Sets up the controller for a new search.
    #[inline(always)]
    pub fn new(pos: &Position, limits: Limits) -> GlobalControl {
        GlobalControl {
            abort: AtomicBool::new(false),
            visited: AtomicU64::new(0),
            time: Self::time_to_search(pos, &limits),
            timestamp: Instant::now(),
            limits,
        }
    }

    /// The search limits.
    #[inline(always)]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The time elapsed so far.
    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        Instant::now()
            .saturating_duration_since(self.timestamp)
            .min(self.limits.max_time())
    }

    /// The nodes visited so far.
    #[inline(always)]
    pub fn visited(&self) -> u64 {
        let max_nodes = self.limits.max_nodes();
        self.visited.load(Ordering::Relaxed).min(max_nodes)
    }

    /// Interrupts an ongoing search.
    #[inline(always)]
    pub fn abort(&self) {
        self.abort.store(true, Ordering::Relaxed);
    }
}

/// The local search control.
#[derive(Debug, Deref)]
pub struct LocalControl<'a> {
    #[deref]
    global: &'a GlobalControl,
    attention: Attention,
    visited: u64,
    nodes: u64,
    trend: f64,
}

impl<'a> LocalControl<'a> {
    const LAP: u64 = 2048;

    #[inline(always)]
    pub fn new(global: &'a GlobalControl) -> Self {
        LocalControl {
            global,
            attention: Default::default(),
            visited: 0,
            nodes: 0,
            trend: f64::NAN,
        }
    }

    /// The PV [`Attention`] statistics.
    #[inline(always)]
    pub fn attention(&mut self) -> &mut Attention {
        &mut self.attention
    }

    /// Whether the search should expand a node.
    #[inline(always)]
    pub fn check(&mut self, ply: Ply, pv: &Pv) -> ControlFlow {
        let score = pv.score().get() as f64;
        let Some(head) = pv.head() else {
            self.nodes += 1;
            return ControlFlow::Continue;
        };

        if self.abort.load(Ordering::Relaxed) {
            return ControlFlow::Abort;
        }

        if self.trend.is_nan() {
            self.trend = score;
        } else if ply == 0 {
            let inertia = Params::score_trend_inertia()[0] as f64 / Params::BASE as f64;
            self.trend = (self.trend * inertia + score) / (inertia + 1.);
        }

        if self.nodes.is_multiple_of(Self::LAP) || ply == 0 {
            let time = self.elapsed().as_secs_f64();
            if time >= self.time.end {
                self.abort.store(true, Ordering::Relaxed);
                return ControlFlow::Abort;
            } else if ply == 0 {
                let gamma = Params::pv_focus_gamma()[0] as f64 / Params::BASE as f64;
                let delta = Params::pv_focus_delta()[0] as f64 / Params::BASE as f64;
                let pivot = Params::score_trend_pivot()[0] as f64 / Params::BASE as f64;
                let magnitude = Params::score_trend_magnitude()[0] as f64 / Params::BASE as f64;

                let nodes = self.nodes.max(1000) as f64;
                let diff = self.trend - pv.score().get() as f64;
                let focus = self.attention.nodes(head).get() as f64 / nodes;
                let scale = 1. + magnitude * diff / (diff.abs() + pivot);
                if time >= self.time.start * scale * (delta - gamma * focus) {
                    return ControlFlow::Stop;
                }
            }
        }

        self.nodes += 1;
        if self.nodes.is_multiple_of(Self::LAP) {
            self.visited = self.global.visited.fetch_add(Self::LAP, Ordering::Relaxed) + Self::LAP;
        }

        let increment = self.nodes % Self::LAP;
        if self.visited + increment >= self.limits.max_nodes() {
            self.global.visited.fetch_add(increment, Ordering::Relaxed);
            self.abort.store(true, Ordering::Relaxed);
            return ControlFlow::Abort;
        }

        ControlFlow::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nnue::Evaluator, search::Score};
    use std::thread;
    use test_strategy::proptest;

    #[proptest]
    fn measures_time_elapsed(pos: Evaluator, l: Limits) {
        let ctrl = GlobalControl::new(&pos, l);
        let duration = Duration::from_millis(1);
        thread::sleep(duration);
        assert!(ctrl.elapsed() >= duration);
    }

    #[proptest]
    fn aborts_if_time_is_up(pos: Evaluator, #[filter(#pv.head().is_some())] pv: Pv) {
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
    }

    #[proptest]
    fn stops_if_searched_for_sufficient_time(
        #[filter(#pos.ply() == 0)] pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let mut global = GlobalControl::new(&pos, Limits::clock(Duration::MAX, Duration::ZERO));
        global.time.start = 0.;

        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Stop);
    }

    #[proptest]
    fn counts_nodes_visited(pos: Evaluator, n: u64, #[filter(#pv.head().is_some())] pv: Pv) {
        let global = GlobalControl::new(&pos, Limits::nodes(n));
        let mut local = LocalControl::new(&global);
        assert_eq!(local.nodes, 0);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.nodes, 1);
    }

    #[proptest]
    fn aborts_if_node_count_is_reached(pos: Evaluator, #[filter(#pv.head().is_some())] pv: Pv) {
        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
    }

    #[proptest]
    fn aborts_upon_request(pos: Evaluator, #[filter(#pv.head().is_some())] pv: Pv) {
        let global = GlobalControl::new(&pos, Limits::none());
        global.abort();

        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(pos.ply(), &pv), ControlFlow::Abort);
    }

    #[proptest]
    fn suspends_limits_while_empty_pv(pos: Evaluator, s: Score) {
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &Pv::empty(s)), ControlFlow::Continue);

        let global = GlobalControl::new(&pos, Limits::clock(Duration::ZERO, Duration::ZERO));
        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &Pv::empty(s)), ControlFlow::Continue);

        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let mut local = LocalControl::new(&global);
        assert_eq!(local.check(pos.ply(), &Pv::empty(s)), ControlFlow::Continue);
    }
}
