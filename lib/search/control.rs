use crate::chess::{Move, Position};
use crate::params::Params;
use crate::search::{Attention, Depth, Limits, Nodes, Ply, Pv};
use crate::util::{Float, Int};
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Controls the search flow.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum ControlFlow {
    /// Continue searching.
    Continue,
    /// Interrupt searching as soon as possible.
    Stop,
    /// Interrupt searching immediately.
    Abort,
}

/// The global search controller.
#[derive(Debug)]
pub struct GlobalControl {
    abort: AtomicBool,
    nodes: AtomicU64,
    timestamp: Instant,
    time: Range<f32>,
    limits: Limits,
}

impl GlobalControl {
    #[inline(always)]
    fn time_to_search(pos: &Position, limits: &Limits) -> Range<f32> {
        let (clock, inc) = match limits.clock {
            Some((clock, inc)) => (clock.as_secs_f32(), inc.as_secs_f32()),
            None => return f32::INFINITY..limits.max_time().as_secs_f32(),
        };

        let time_left = clock - inc;
        let moves_left_start = *Params::moves_left_start(0);
        let moves_left_end = *Params::moves_left_end(0);
        let max_fullmoves = moves_left_start / moves_left_end;
        let moves_left = moves_left_start / max_fullmoves.min(pos.fullmoves().get() as f32);
        let time_per_move = inc + time_left / moves_left;

        let soft_time_fraction = *Params::soft_time_fraction(0);
        let hard_time_fraction = *Params::hard_time_fraction(0);
        soft_time_fraction * time_per_move..clock * hard_time_fraction
    }

    /// Sets up the controller for a new search.
    #[inline(always)]
    pub fn new(pos: &Position, limits: Limits) -> GlobalControl {
        GlobalControl {
            abort: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            timestamp: Instant::now(),
            time: Self::time_to_search(pos, &limits),
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
        self.timestamp.elapsed()
    }

    /// The number of nodes visited so far.
    #[inline(always)]
    pub fn visited(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }

    /// Interrupts an ongoing search.
    #[inline(always)]
    pub fn abort(&self) {
        self.abort.store(true, Ordering::Relaxed);
    }
}

/// The active search controller.
#[derive(Debug)]
pub struct Active<'a> {
    global: &'a GlobalControl,
    attention: Attention,
    peak_depth: Depth,
    score_trend: f32,
    global_nodes: u64,
    nodes: u64,
}

impl<'a> Active<'a> {
    #[inline(always)]
    pub fn new(global: &'a GlobalControl) -> Self {
        Active {
            global,
            attention: Default::default(),
            peak_depth: Depth::lower(),
            score_trend: f32::NAN,
            global_nodes: 0,
            nodes: 0,
        }
    }

    #[inline(always)]
    pub fn check(&mut self, depth: Depth, ply: Ply, pv: &Pv) -> ControlFlow {
        let score = pv.score().get() as f32;
        let Some(head) = pv.head() else {
            return ControlFlow::Continue;
        };

        if self.abort.load(Ordering::Relaxed) {
            return ControlFlow::Abort;
        }

        if self.score_trend.is_nan() {
            self.score_trend = score;
        } else if ply == 0 && depth > self.peak_depth {
            self.score_trend = Params::score_trend_inertia(0).lerp(self.score_trend, score);
            self.peak_depth = depth;
        }

        if self.nodes.is_multiple_of(2048) || ply == 0 {
            let time = self.elapsed().as_secs_f32();
            if time >= self.time.end {
                return ControlFlow::Abort;
            } else if ply == 0 {
                let gamma = *Params::pv_focus_gamma(0);
                let delta = *Params::pv_focus_delta(0);
                let pivot = *Params::score_trend_pivot(0);
                let magnitude = *Params::score_trend_magnitude(0);

                let nodes = self.nodes.max(1000) as f32;
                let diff = self.score_trend - pv.score().get() as f32;
                let focus = self.attention.nodes(head).get() as f32 / nodes;
                let scale = 1. + magnitude * diff / (diff.abs() + pivot);
                if time >= self.time.start * scale * focus.mul_add(gamma, delta) {
                    return ControlFlow::Stop;
                }
            }
        }

        if depth > self.limits.max_depth() {
            return ControlFlow::Stop;
        }

        self.nodes += 1;
        const LAP: u64 = 1024;
        if self.nodes.is_multiple_of(LAP) {
            self.global_nodes = self.global.nodes.fetch_add(LAP, Ordering::Relaxed) + LAP;
        }

        if self.global_nodes + self.nodes % LAP >= self.limits.max_nodes() {
            return ControlFlow::Stop;
        }

        ControlFlow::Continue
    }
}

impl const Deref for Active<'_> {
    type Target = GlobalControl;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.global
    }
}

/// The passive search controller.
#[derive(Debug)]
pub struct Passive<'a> {
    global: &'a GlobalControl,
    nodes: u64,
}

impl<'a> Passive<'a> {
    #[inline(always)]
    pub fn new(global: &'a GlobalControl) -> Self {
        Passive { global, nodes: 0 }
    }

    #[inline(always)]
    pub fn check(&mut self) -> ControlFlow {
        if self.abort.load(Ordering::Relaxed) {
            return ControlFlow::Abort;
        }

        self.nodes += 1;
        const LAP: u64 = 16384;
        if self.nodes.is_multiple_of(LAP) {
            self.global.nodes.fetch_add(LAP, Ordering::Relaxed);
        }

        ControlFlow::Continue
    }
}

impl const Deref for Passive<'_> {
    type Target = GlobalControl;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.global
    }
}

/// The local search controller.
#[derive(Debug)]
#[expect(clippy::large_enum_variant)]
pub enum LocalControl<'a> {
    Active(Active<'a>),
    Passive(Passive<'a>),
}

impl const Deref for LocalControl<'_> {
    type Target = GlobalControl;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            LocalControl::Active(ctrl) => ctrl,
            LocalControl::Passive(ctrl) => ctrl,
        }
    }
}

impl<'a> LocalControl<'a> {
    #[inline(always)]
    pub fn active(global: &'a GlobalControl) -> Self {
        LocalControl::Active(Active::new(global))
    }

    #[inline(always)]
    pub fn passive(global: &'a GlobalControl) -> Self {
        LocalControl::Passive(Passive::new(global))
    }

    /// The PV [`Attention`] statistics.
    #[inline(always)]
    pub fn attention(&mut self, head: Move) -> Option<NonNull<Nodes>> {
        match self {
            LocalControl::Active(ctrl) => Some(NonNull::from_mut(ctrl.attention.nodes(head))),
            LocalControl::Passive(_) => None,
        }
    }

    /// Whether the search should expand a node.
    #[inline(always)]
    pub fn check(&mut self, depth: Depth, ply: Ply, pv: &Pv) -> ControlFlow {
        match self {
            LocalControl::Active(ctrl) => ctrl.check(depth, ply, pv),
            LocalControl::Passive(ctrl) => ctrl.check(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nnue::Evaluator, search::Score};
    use std::thread;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn global_measures_time_elapsed(pos: Evaluator, l: Limits) {
        let ctrl = GlobalControl::new(&pos, l);
        let duration = Duration::from_millis(1);
        thread::sleep(duration);
        assert!(ctrl.elapsed() >= duration);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_counts_nodes_visited(
        #[filter(#d < Depth::MAX)] d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::none());
        let mut active = Active::new(&global);
        assert_eq!(active.nodes, 0);
        assert_eq!(active.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(active.nodes, 1);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn passive_counts_nodes_visited(pos: Evaluator) {
        let global = GlobalControl::new(&pos, Limits::none());
        let mut passive = Passive::new(&global);
        assert_eq!(passive.nodes, 0);
        assert_eq!(passive.check(), ControlFlow::Continue);
        assert_eq!(passive.nodes, 1);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_aborts_if_time_is_up(
        pos: Evaluator,
        d: Depth,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn passive_continues_if_time_is_up(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let mut local = LocalControl::passive(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_stops_if_searched_for_sufficient_time(
        d: Depth,
        #[filter(#pos.ply() == 0)] pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let mut global = GlobalControl::new(&pos, Limits::clock(Duration::MAX, Duration::ZERO));
        global.time.start = 0.;

        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Stop);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn passive_continues_if_searched_for_sufficient_time(
        d: Depth,
        #[filter(#pos.ply() == 0)] pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let mut global = GlobalControl::new(&pos, Limits::clock(Duration::MAX, Duration::ZERO));
        global.time.start = 0.;

        let mut local = LocalControl::passive(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_stops_if_target_depth_is_reached(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::depth(Depth::lower()));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d + 1, pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(d + 1, pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(d + 1, pos.ply(), &pv), ControlFlow::Stop);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn passive_continues_if_target_depth_is_reached(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::depth(Depth::lower()));
        let mut local = LocalControl::passive(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_stops_if_node_count_is_reached(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Stop);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Stop);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn passive_continues_if_node_count_is_reached(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let mut local = LocalControl::passive(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_aborts_upon_request(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::none());
        global.abort();

        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn passive_aborts_upon_request(
        d: Depth,
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let global = GlobalControl::new(&pos, Limits::none());
        global.abort();

        let mut local = LocalControl::passive(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Abort);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn active_suspends_limits_while_empty_pv(d: Depth, pos: Evaluator, s: Score) {
        let pv = Pv::empty(s);

        let global = GlobalControl::new(&pos, Limits::time(Duration::ZERO));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);

        let global = GlobalControl::new(&pos, Limits::clock(Duration::ZERO, Duration::ZERO));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);

        let global = GlobalControl::new(&pos, Limits::depth(Depth::lower()));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);

        let global = GlobalControl::new(&pos, Limits::nodes(0));
        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);

        let global = GlobalControl::new(&pos, Limits::nodes(0));
        global.abort();

        let mut local = LocalControl::active(&global);
        assert_eq!(local.check(d, pos.ply(), &pv), ControlFlow::Continue);
    }
}
