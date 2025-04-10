use crate::chess::{Butterfly, Move, Position};
use crate::search::{Attention, Limits, Statistics};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::{mem::MaybeUninit, ops::Range};

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
    time: Range<Duration>,
    timestamp: Instant,
    abort: AtomicBool,
    stop: Butterfly<AtomicBool>,
}

impl Control {
    #[inline(always)]
    fn time_to_search(pos: &Position, limits: &Limits) -> Range<Duration> {
        let Limits::Clock(clock, inc) = *limits else {
            return limits.time()..limits.time();
        };

        let time_left = clock.saturating_sub(inc);
        let moves_left = 225 / pos.fullmoves().get().min(75);
        let time_per_move = inc.saturating_add(time_left / moves_left);
        time_per_move / 2..clock / 2
    }

    /// Sets up the controller for a new search.
    #[inline(always)]
    pub fn new(pos: &Position, limits: Limits) -> Control {
        Control {
            nodes: AtomicU64::new(limits.nodes()),
            attention: Attention::default(),
            time: Self::time_to_search(pos, &limits),
            timestamp: Instant::now(),
            limits,
            abort: AtomicBool::new(false),
            stop: unsafe { MaybeUninit::zeroed().assume_init() },
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
    pub fn check(&self, root: &Position, m: Move) -> ControlFlow {
        use Ordering::Relaxed;

        if self.abort.load(Relaxed) {
            return ControlFlow::Abort;
        }

        let checked_dec = |i: u64| i.checked_sub(1);
        let nodes = match self.nodes.fetch_update(Relaxed, Relaxed, checked_dec) {
            Ok(count) => self.limits.nodes() - count,
            Err(_) => {
                self.abort.store(true, Relaxed);
                return ControlFlow::Abort;
            }
        };

        let stop = &self.stop[m.whence() as usize][m.whither() as usize];
        let focus = match self.limits {
            Limits::Clock(..) => self.attention.get(root, m) as f64 / nodes.max(1024) as f64,
            _ => 0.,
        };

        if nodes % 1024 == 0 {
            let time = self.time();
            if time >= self.time.end {
                self.abort.store(true, Relaxed);
                return ControlFlow::Abort;
            } else if time.as_secs_f64() > self.time.start.as_secs_f64() * (3. - 2.8 * focus) {
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
    fn aborts_if_time_is_up(pos: Position, m: Move) {
        let ctrl = Control::new(&pos, Limits::Time(Duration::ZERO));
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
    }

    #[proptest]
    fn stops_if_searched_for_sufficient_time(pos: Position, m: Move) {
        let mut ctrl = Control::new(&pos, Limits::Time(Duration::MAX));
        ctrl.time.start = Duration::ZERO;
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Stop);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Stop);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Stop);
    }

    #[proptest]
    fn counts_nodes_searched(pos: Position, n: u64, m: Move) {
        let ctrl = Control::new(&pos, Limits::Nodes(n));
        assert_eq!(ctrl.nodes(), 0);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Continue);
        assert_eq!(ctrl.nodes(), 1);
    }

    #[proptest]
    fn aborts_if_node_count_is_reached(pos: Position, m: Move) {
        let ctrl = Control::new(&pos, Limits::Nodes(0));
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
    }

    #[proptest]
    fn aborts_upon_request(pos: Position, m: Move) {
        let ctrl = Control::new(&pos, Limits::None);
        ctrl.abort();
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
        assert_eq!(ctrl.check(&pos, m), ControlFlow::Abort);
    }
}
