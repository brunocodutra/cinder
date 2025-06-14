use crate::chess::{Butterfly, Position};
use crate::nnue::Evaluator;
use crate::search::{Attention, Limits, Ply, Pv, Statistics};
use crate::{params::Params, util::Integer};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::{array::from_fn, ops::Range};

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

/// The search control.
#[derive(Debug)]
pub struct Control {
    limits: Limits,
    nodes: AtomicU64,
    attention: Attention,
    time: Range<f64>,
    timestamp: Instant,
    trend: AtomicU64,
    abort: AtomicBool,
    stop: Butterfly<AtomicBool>,
}

impl Control {
    #[inline(always)]
    fn time_to_search(pos: &Position, limits: &Limits) -> Range<f64> {
        let (clock, inc) = match limits.clock {
            Some((clock, inc)) => (clock.as_secs_f64(), inc.as_secs_f64()),
            None => return f64::INFINITY..limits.max_time().as_secs_f64(),
        };

        let time_left = clock - inc;
        let moves_left_start = Params::moves_left_start() as f64 / Params::BASE as f64;
        let moves_left_end = Params::moves_left_end() as f64 / Params::BASE as f64;
        let max_fullmoves = moves_left_start / moves_left_end;
        let moves_left = moves_left_start / max_fullmoves.min(pos.fullmoves().get() as _);
        let time_per_move = inc + time_left / moves_left;

        let soft_time_fraction = Params::soft_time_fraction() as f64 / Params::BASE as f64;
        let hard_time_fraction = Params::hard_time_fraction() as f64 / Params::BASE as f64;
        soft_time_fraction * time_per_move..clock * hard_time_fraction
    }

    /// Sets up the controller for a new search.
    #[inline(always)]
    pub fn new(pos: &Position, limits: Limits) -> Control {
        Control {
            nodes: AtomicU64::new(limits.max_nodes()),
            attention: Attention::default(),
            time: Self::time_to_search(pos, &limits),
            timestamp: Instant::now(),
            trend: AtomicU64::new(f64::NAN.to_bits()),
            abort: AtomicBool::new(false),
            stop: from_fn(|_| from_fn(|_| AtomicBool::new(false))),
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
    pub fn time(&self) -> Duration {
        Instant::now()
            .saturating_duration_since(self.timestamp)
            .max(Duration::from_nanos(1))
    }

    /// The nodes counted so far.
    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        self.limits.max_nodes() - self.nodes.load(Ordering::Relaxed)
    }

    /// The PV [`Attention`] statistics.
    #[inline(always)]
    pub fn attention(&self) -> &Attention {
        &self.attention
    }

    /// Interrupts an ongoing search.
    #[inline(always)]
    pub fn abort(&self) {
        self.abort.store(true, Ordering::Relaxed);
    }

    /// Whether the search should expand a node.
    #[inline(always)]
    pub fn check(&self, evaluator: &Evaluator, pv: &Pv) -> ControlFlow {
        use Ordering::Relaxed;

        if self.abort.load(Relaxed) {
            return ControlFlow::Abort;
        }

        let Some(best) = pv.head() else {
            return ControlFlow::Continue;
        };

        let checked_dec = |i: u64| i.checked_sub(1);
        let nodes = match self.nodes.fetch_update(Relaxed, Relaxed, checked_dec) {
            Ok(count) => self.limits.max_nodes() - count,
            Err(_) => {
                self.abort.store(true, Relaxed);
                return ControlFlow::Abort;
            }
        };

        let ply = evaluator.ply();
        let inertia = Params::score_trend_inertia() as f64 / Params::BASE as f64;
        let _ = self.trend.fetch_update(Relaxed, Relaxed, |bits| {
            let score = pv.score().get() as f64;
            match f64::from_bits(bits) {
                s if s.is_nan() => Some(score.to_bits()),
                s if ply == 0 => Some(((s * inertia + score) / (inertia + 1.)).to_bits()),
                _ => None,
            }
        });

        let stop = &self.stop[best.whence() as usize][best.whither() as usize];
        let focus_gamma = Params::pv_focus_gamma() as f64 / Params::BASE as f64;
        let focus_delta = Params::pv_focus_delta() as f64 / Params::BASE as f64;
        let trend_magnitude = Params::score_trend_magnitude() as f64 / Params::BASE as f64;
        let trend_pivot = Params::score_trend_pivot() as f64 / Params::BASE as f64;

        if nodes % 1024 == 0 {
            let root = &evaluator[Ply::new(0)];
            let time = self.time().as_secs_f64();
            let focus = self.attention.get(root, best) as f64 / nodes.max(1024) as f64;
            let delta = f64::from_bits(self.trend.load(Relaxed)) - pv.score().get() as f64;
            let focus_scale = focus_delta - focus_gamma * focus;
            let trend_scale = 1. + trend_magnitude * delta / (delta.abs() + trend_pivot);
            let soft_time_scale = focus_scale * trend_scale;

            if time >= self.time.end {
                self.abort.store(true, Relaxed);
                return ControlFlow::Abort;
            } else if time >= self.time.start * soft_time_scale {
                stop.store(true, Relaxed);
                return ControlFlow::Stop;
            }
        }

        if stop.load(Relaxed) {
            ControlFlow::Stop
        } else {
            ControlFlow::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Score;
    use std::thread;
    use test_strategy::proptest;

    #[proptest]
    fn measures_time_elapsed(pos: Evaluator, l: Limits) {
        let ctrl = Control::new(&pos, l);
        let duration = Duration::from_millis(1);
        thread::sleep(duration);
        assert!(ctrl.time() >= duration);
    }

    #[proptest]
    fn time_elapsed_is_always_positive(pos: Evaluator, l: Limits) {
        let ctrl = Control::new(&pos, l);
        assert!(ctrl.time() > Duration::ZERO);
    }

    #[proptest]
    fn aborts_if_time_is_up(pos: Evaluator, #[filter(#pv.head().is_some())] pv: Pv) {
        let ctrl = Control::new(&pos, Limits::time(Duration::ZERO));
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
    }

    #[proptest]
    fn stops_if_searched_for_sufficient_time(
        pos: Evaluator,
        #[filter(#pv.head().is_some())] pv: Pv,
    ) {
        let mut ctrl = Control::new(&pos, Limits::clock(Duration::MAX, Duration::ZERO));
        ctrl.time.start = 0.;
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Stop);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Stop);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Stop);
    }

    #[proptest]
    fn counts_nodes_searched(pos: Evaluator, n: u64, #[filter(#pv.head().is_some())] pv: Pv) {
        let ctrl = Control::new(&pos, Limits::nodes(n));
        assert_eq!(ctrl.nodes(), 0);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Continue);
        assert_eq!(ctrl.nodes(), 1);
    }

    #[proptest]
    fn aborts_if_node_count_is_reached(pos: Evaluator, #[filter(#pv.head().is_some())] pv: Pv) {
        let ctrl = Control::new(&pos, Limits::nodes(0));
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
    }

    #[proptest]
    fn aborts_upon_request(pos: Evaluator, #[filter(#pv.head().is_some())] pv: Pv) {
        let ctrl = Control::new(&pos, Limits::none());
        ctrl.abort();
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, &pv), ControlFlow::Abort);
    }

    #[proptest]
    fn suspends_limits_while_empty_pv(pos: Evaluator, s: Score) {
        let ctrl = Control::new(&pos, Limits::time(Duration::ZERO));
        assert_eq!(ctrl.check(&pos, &Pv::empty(s)), ControlFlow::Continue);

        let ctrl = Control::new(&pos, Limits::clock(Duration::ZERO, Duration::ZERO));
        assert_eq!(ctrl.check(&pos, &Pv::empty(s)), ControlFlow::Continue);

        let ctrl = Control::new(&pos, Limits::nodes(0));
        assert_eq!(ctrl.check(&pos, &Pv::empty(s)), ControlFlow::Continue);
    }
}
