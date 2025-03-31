use crate::{search::Depth, util::Integer};
use derive_more::with_trait::From;
use std::time::Duration;

/// Configuration for search limits.
#[derive(Debug, Default, Clone, Eq, PartialEq, From)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum Limits {
    /// Unlimited search.
    #[default]
    None,

    /// The maximum number of plies to search.
    Depth(Depth),

    /// The maximum number of nodes to search.
    Nodes(u64),

    /// The maximum amount of time to spend searching.
    Time(Duration),

    /// The time remaining on the clock.
    #[from(ignore)]
    Clock(Duration, Duration),
}

impl Limits {
    /// Maximum depth or [`Depth::MAX`].
    #[inline(always)]
    pub fn depth(&self) -> Depth {
        match self {
            Limits::Depth(d) => *d,
            _ => Depth::upper(),
        }
    }

    /// Maximum number of nodes [`u64::MAX`].
    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        match self {
            Limits::Nodes(n) => *n,
            _ => u64::MAX,
        }
    }

    /// Maximum time or [`Duration::MAX`].
    #[inline(always)]
    pub fn time(&self) -> Duration {
        match self {
            Limits::Time(t) => *t,
            Limits::Clock(t, _) => *t,
            _ => Duration::MAX,
        }
    }

    /// Time left on the clock or [`Duration::MAX`].
    #[inline(always)]
    pub fn clock(&self) -> Duration {
        match self {
            Limits::Clock(t, _) => *t,
            _ => Duration::MAX,
        }
    }

    /// Time increment or [`Duration::ZERO`].
    #[inline(always)]
    pub fn increment(&self) -> Duration {
        match self {
            Limits::Clock(_, i) => *i,
            _ => Duration::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn depth_returns_value_if_set(d: Depth) {
        assert_eq!(Limits::Depth(d).depth(), d);
    }

    #[proptest]
    fn depth_returns_max_by_default(n: u64, t: Duration, i: Duration) {
        assert_eq!(Limits::None.depth(), Depth::MAX);
        assert_eq!(Limits::Nodes(n).depth(), Depth::MAX);
        assert_eq!(Limits::Time(t).depth(), Depth::MAX);
        assert_eq!(Limits::Clock(t, i).depth(), Depth::MAX);
    }

    #[proptest]
    fn nodes_returns_value_if_set(n: u64) {
        assert_eq!(Limits::Nodes(n).nodes(), n);
    }

    #[proptest]
    fn nodes_returns_max_by_default(d: Depth, t: Duration, i: Duration) {
        assert_eq!(Limits::None.nodes(), u64::MAX);
        assert_eq!(Limits::Depth(d).nodes(), u64::MAX);
        assert_eq!(Limits::Time(t).nodes(), u64::MAX);
        assert_eq!(Limits::Clock(t, i).nodes(), u64::MAX);
    }

    #[proptest]
    fn time_returns_value_if_set(t: Duration) {
        assert_eq!(Limits::Time(t).time(), t);
    }

    #[proptest]
    fn time_returns_max_or_clock_by_default(d: Depth, n: u64, t: Duration, i: Duration) {
        assert_eq!(Limits::None.time(), Duration::MAX);
        assert_eq!(Limits::Depth(d).time(), Duration::MAX);
        assert_eq!(Limits::Nodes(n).time(), Duration::MAX);
        assert_eq!(Limits::Clock(t, i).time(), t);
    }

    #[proptest]
    fn clock_returns_value_if_set(t: Duration, i: Duration) {
        assert_eq!(Limits::Clock(t, i).clock(), t);
    }

    #[proptest]
    fn clock_returns_max_by_default(d: Depth, n: u64, t: Duration) {
        assert_eq!(Limits::None.clock(), Duration::MAX);
        assert_eq!(Limits::Depth(d).clock(), Duration::MAX);
        assert_eq!(Limits::Nodes(n).clock(), Duration::MAX);
        assert_eq!(Limits::Time(t).clock(), Duration::MAX);
    }

    #[proptest]
    fn increment_returns_value_if_set(t: Duration, i: Duration) {
        assert_eq!(Limits::Clock(t, i).increment(), i);
    }

    #[proptest]
    fn increment_returns_zero_by_default(d: Depth, n: u64, t: Duration) {
        assert_eq!(Limits::None.increment(), Duration::ZERO);
        assert_eq!(Limits::Depth(d).increment(), Duration::ZERO);
        assert_eq!(Limits::Nodes(n).increment(), Duration::ZERO);
        assert_eq!(Limits::Time(t).increment(), Duration::ZERO);
    }
}
