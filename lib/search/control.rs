use crate::{chess::Position, search::Limits, util::Integer};
use std::ops::Range;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Controls the search flow.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum ControlFlow {
    /// Continue searching.
    Continue,
    /// Interrupt searching as soon as possible.
    Stop,
    /// Interrupt searching immediately.
    Abort,
}

unsafe impl Integer for ControlFlow {
    type Repr = u8;
    const MIN: Self::Repr = ControlFlow::Continue as _;
    const MAX: Self::Repr = ControlFlow::Abort as _;
}

/// The search control.
#[derive(Debug)]
pub struct Control {
    limits: Limits,
    flow: AtomicU8,
    nodes: AtomicU64,
    time: Range<Duration>,
    timestamp: Instant,
}

impl Control {
    #[inline(always)]
    fn time_to_search(pos: &Position, limits: &Limits) -> Range<Duration> {
        let Limits::Clock(clock, inc) = *limits else {
            return limits.time()..limits.time();
        };

        let time_left = clock.saturating_sub(inc);
        let moves_left = 256 / pos.fullmoves().get().min(64);
        let time_per_move = inc.saturating_add(time_left / moves_left).min(clock / 2);
        time_per_move / 2..time_per_move
    }

    /// Sets up the controller for a new search.
    #[inline(always)]
    pub fn new(pos: &Position, limits: Limits) -> Control {
        Control {
            flow: AtomicU8::new(ControlFlow::Continue.get()),
            nodes: AtomicU64::new(limits.nodes()),
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
    pub fn time(&self) -> Duration {
        Instant::now()
            .saturating_duration_since(self.timestamp)
            .max(Duration::from_nanos(1))
    }

    /// The nodes counted so far.
    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        self.limits.nodes() - self.nodes.load(Ordering::Relaxed)
    }

    /// Interrupts an ongoing search.
    #[inline(always)]
    pub fn abort(&self) {
        use {ControlFlow::*, Ordering::Relaxed};
        self.flow.store(Abort.get(), Relaxed);
    }

    /// Whether the search should expand a node.
    #[inline(always)]
    pub fn check(&self) -> ControlFlow {
        use Ordering::Relaxed;

        if ControlFlow::new(self.flow.load(Relaxed)) == ControlFlow::Abort {
            return ControlFlow::Abort;
        }

        let checked_dec = |i: u64| i.checked_sub(1);
        let nodes = match self.nodes.fetch_update(Relaxed, Relaxed, checked_dec) {
            Ok(count) => self.limits.nodes() - count,
            Err(_) => {
                self.flow.store(ControlFlow::Abort.get(), Relaxed);
                return ControlFlow::Abort;
            }
        };

        if nodes % 1024 == 0 {
            let time = self.time();
            if time >= self.time.end {
                self.flow.store(ControlFlow::Abort.get(), Relaxed);
                return ControlFlow::Abort;
            } else if time >= self.time.start {
                self.flow.store(ControlFlow::Stop.get(), Relaxed);
                return ControlFlow::Stop;
            }
        }

        ControlFlow::new(self.flow.load(Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use test_strategy::proptest;

    #[proptest]
    fn measures_time_elapsed(pos: Position, l: Limits) {
        let ctrl = Control::new(&pos, l);
        let duration = Duration::from_millis(1);
        thread::sleep(duration);
        assert!(ctrl.time() >= duration);
    }

    #[proptest]
    fn time_elapsed_is_always_positive(pos: Position, l: Limits) {
        let ctrl = Control::new(&pos, l);
        assert!(ctrl.time() > Duration::ZERO);
    }

    #[proptest]
    fn aborts_if_time_is_up(pos: Position) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        assert_eq!(ctrl.check(), ControlFlow::Abort);
        assert_eq!(ctrl.check(), ControlFlow::Abort);
        assert_eq!(ctrl.check(), ControlFlow::Abort);
    }

    #[proptest]
    fn stops_if_searched_for_sufficient_time(pos: Position) {
        let mut ctrl = Control::new(&pos, Limits::Time(Duration::MAX));
        ctrl.time.start = Duration::ZERO;
        assert_eq!(ctrl.check(), ControlFlow::Stop);
        assert_eq!(ctrl.check(), ControlFlow::Stop);
        assert_eq!(ctrl.check(), ControlFlow::Stop);
    }

    #[proptest]
    fn counts_nodes_searched(pos: Position, n: u64) {
        let ctrl = Control::new(&pos, Limits::Nodes(n));
        assert_eq!(ctrl.nodes(), 0);
        assert_eq!(ctrl.check(), ControlFlow::Continue);
        assert_eq!(ctrl.nodes(), 1);
    }

    #[proptest]
    fn aborts_if_node_count_is_reached(pos: Position) {
        let ctrl = Control::new(&pos, Limits::Nodes(0));
        assert_eq!(ctrl.check(), ControlFlow::Abort);
        assert_eq!(ctrl.check(), ControlFlow::Abort);
        assert_eq!(ctrl.check(), ControlFlow::Abort);
    }

    #[proptest]
    fn aborts_upon_request(pos: Position) {
        let ctrl = Control::new(&pos, Limits::None);
        ctrl.abort();
        assert_eq!(ctrl.check(), ControlFlow::Abort);
        assert_eq!(ctrl.check(), ControlFlow::Abort);
        assert_eq!(ctrl.check(), ControlFlow::Abort);
    }
}
