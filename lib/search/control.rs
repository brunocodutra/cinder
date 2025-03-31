use crate::{chess::Position, search::Limits};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Controls the search flow.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ControlFlow {
    /// Continue searching.
    Continue,
    /// Stop searching as soon as possible.
    Stop,
    /// Stop searching immediately.
    Interrupt,
}

/// The search control.
#[derive(Debug)]
pub struct Control {
    limits: Limits,
    interrupted: AtomicBool,
    nodes: AtomicU64,
    time: Range<Duration>,
    timer: Instant,
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
            interrupted: AtomicBool::new(false),
            nodes: AtomicU64::new(limits.nodes()),
            time: Self::time_to_search(pos, &limits),
            timer: Instant::now(),
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
            .saturating_duration_since(self.timer)
            .max(Duration::from_nanos(1))
    }

    /// The nodes counted so far.
    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        self.limits.nodes() - self.nodes.load(Ordering::Relaxed)
    }

    /// Interrupts an ongoing search.
    ///
    /// Returns `true` if an ongoing search was interrupted.
    pub fn interrupt(&self) -> bool {
        !self.interrupted.fetch_or(true, Ordering::Relaxed)
    }

    /// Whether the search should expand a node.
    #[inline(always)]
    pub fn check(&self) -> ControlFlow {
        use Ordering::Relaxed;
        if self.interrupted.load(Relaxed) {
            return ControlFlow::Interrupt;
        }

        let checked_dec = |i: u64| i.checked_sub(1);
        let nodes = match self.nodes.fetch_update(Relaxed, Relaxed, checked_dec) {
            Ok(count) => self.limits.nodes() - count,
            Err(_) => return ControlFlow::Interrupt,
        };

        if nodes % 256 > 0 {
            return ControlFlow::Continue;
        }

        let time = self.time();
        if time >= self.time.end {
            ControlFlow::Interrupt
        } else if time >= self.time.start {
            ControlFlow::Stop
        } else {
            ControlFlow::Continue
        }
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
    fn interrupts_if_time_is_up(pos: Position) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        assert_eq!(ctrl.check(), ControlFlow::Interrupt);
    }

    #[proptest]
    fn counts_nodes_searched(pos: Position, n: u64) {
        let ctrl = Control::new(&pos, Limits::Nodes(n));
        assert_eq!(ctrl.nodes(), 0);
        assert_eq!(ctrl.check(), ControlFlow::Continue);
        assert_eq!(ctrl.nodes(), 1);
    }

    #[proptest]
    fn interrupts_if_node_count_is_reached(pos: Position) {
        let ctrl = Control::new(&pos, Limits::Nodes(0));
        assert_eq!(ctrl.check(), ControlFlow::Interrupt);
    }

    #[proptest]
    fn interrupts_upon_request(pos: Position) {
        let ctrl = Control::new(&pos, Limits::None);
        assert!(ctrl.interrupt());
        assert_eq!(ctrl.check(), ControlFlow::Interrupt);
    }

    #[proptest]
    fn can_only_be_interrupted_once(pos: Position) {
        let ctrl = Control::new(&pos, Limits::None);
        assert!(ctrl.interrupt());
        assert!(!ctrl.interrupt());
        assert!(!ctrl.interrupt());
    }
}
