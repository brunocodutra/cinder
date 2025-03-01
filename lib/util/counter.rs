use std::sync::atomic::{AtomicU64, Ordering};

/// A counter towards a limit.
#[derive(Debug)]
pub struct Counter {
    remaining: AtomicU64,
}

impl Counter {
    /// Constructs a counter with the given limit.
    #[inline(always)]
    pub const fn new(limit: u64) -> Self {
        Counter {
            remaining: AtomicU64::new(limit),
        }
    }

    /// The number of counts remaining.
    #[inline(always)]
    pub fn get(&self) -> u64 {
        self.remaining.load(Ordering::Relaxed)
    }

    /// Increments the counter and returns the number of counts remaining if any.
    #[inline(always)]
    pub fn count(&self) -> Option<u64> {
        self.remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| c.checked_sub(1))
            .map_or(None, |c| Some(c - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn counter_keeps_track_of_counts_remaining(#[strategy(1u64..)] c: u64) {
        let counter = Counter::new(c);
        assert_eq!(counter.count(), Some(c - 1));
    }

    #[test]
    fn counter_stops_once_limit_is_reached() {
        let counter = Counter::new(0);
        assert_eq!(counter.count(), None);
    }
}
