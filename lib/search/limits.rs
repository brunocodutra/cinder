use crate::{search::Depth, util::Integer};
use std::time::Duration;

/// Search limits that can be applied simultaneously.
///
/// The search stops when the first limit is reached.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Limits {
    /// The maximum number of plies to search.
    pub depth: Option<Depth>,

    /// The maximum number of nodes to search.
    pub nodes: Option<u64>,

    /// The maximum amount of time to spend searching.
    pub time: Option<Duration>,

    /// The time remaining on the clock (time, increment).
    pub clock: Option<(Duration, Duration)>,
}

impl Limits {
    /// Create unlimited search limits.
    #[inline(always)]
    pub fn none() -> Self {
        Self::default()
    }

    /// Create limits with only depth restriction.
    #[inline(always)]
    pub fn depth(depth: Depth) -> Self {
        Self {
            depth: Some(depth),
            ..Default::default()
        }
    }

    /// Create limits with only node count restriction.
    #[inline(always)]
    pub fn nodes(nodes: u64) -> Self {
        Self {
            nodes: Some(nodes),
            ..Default::default()
        }
    }

    /// Create limits with only time restriction.
    #[inline(always)]
    pub fn time(time: Duration) -> Self {
        Self {
            time: Some(time),
            ..Default::default()
        }
    }

    /// Create limits with only clock restriction.
    #[inline(always)]
    pub fn clock(time: Duration, increment: Duration) -> Self {
        Self {
            clock: Some((time, increment)),
            ..Default::default()
        }
    }

    /// Get the effective maximum depth.
    ///
    /// Returns the set depth or [`Depth::MAX`] if unlimited.
    #[inline(always)]
    pub fn max_depth(&self) -> Depth {
        self.depth.unwrap_or_else(Depth::upper)
    }

    /// Get the effective maximum number of nodes.
    ///
    /// Returns the set node limit or [`u64::MAX`] if unlimited.
    #[inline(always)]
    pub fn max_nodes(&self) -> u64 {
        self.nodes.unwrap_or_else(u64::upper)
    }

    /// Get the effective maximum time.
    /// Returns the set time limit, clock time, or [`Duration::MAX`] if unlimited.
    #[inline(always)]
    pub fn max_time(&self) -> Duration {
        let time = self.time.unwrap_or(Duration::MAX);
        let clock = self.clock.map_or(Duration::MAX, |(t, _)| t);
        Duration::min(time, clock)
    }

    /// Set depth limit.
    #[must_use]
    #[inline(always)]
    pub fn with_depth(mut self, depth: Depth) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Set node limit.
    #[must_use]
    #[inline(always)]
    pub fn with_nodes(mut self, nodes: u64) -> Self {
        self.nodes = Some(nodes);
        self
    }

    /// Set time limit.
    #[must_use]
    #[inline(always)]
    pub fn with_time(mut self, time: Duration) -> Self {
        self.time = Some(time);
        self
    }

    /// Set clock limit.
    #[must_use]
    #[inline(always)]
    pub fn with_clock(mut self, time: Duration, increment: Duration) -> Self {
        self.clock = Some((time, increment));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[test]
    fn default_is_unlimited() {
        assert_eq!(Limits::default(), Limits::none());
    }

    #[test]
    fn none_is_unlimited() {
        let limits = Limits::none();

        assert_eq!(limits.max_depth(), Depth::upper());
        assert_eq!(limits.max_nodes(), u64::MAX);
        assert_eq!(limits.max_time(), Duration::MAX);
    }

    #[proptest]
    fn can_combine_multiple_limits(d: Depth, n: u64, t: Duration, i: Duration) {
        let limits = Limits::none()
            .with_depth(d)
            .with_nodes(n)
            .with_time(t)
            .with_clock(t, i);

        assert_eq!(limits.max_depth(), d);
        assert_eq!(limits.max_nodes(), n);
        assert_eq!(limits.max_time(), t);
    }

    #[proptest]
    fn depth_returns_value_if_set(d: Depth) {
        assert_eq!(Limits::depth(d).max_depth(), d);
    }

    #[proptest]
    fn depth_returns_max_by_default(n: u64, t: Duration, i: Duration) {
        let limits = Limits::none().with_nodes(n).with_time(t).with_clock(t, i);
        assert_eq!(limits.max_depth(), Depth::upper());
    }

    #[proptest]
    fn nodes_returns_value_if_set(n: u64) {
        assert_eq!(Limits::nodes(n).max_nodes(), n);
    }

    #[proptest]
    fn nodes_returns_max_by_default(d: Depth, t: Duration, i: Duration) {
        let limits = Limits::none().with_depth(d).with_time(t).with_clock(t, i);
        assert_eq!(limits.max_nodes(), u64::MAX);
    }

    #[proptest]
    fn time_returns_min_of_time_or_clock_if_set(t: Duration, u: Duration, i: Duration) {
        assert_eq!(Limits::time(t).with_clock(u, i).max_time(), t.min(u));
    }

    #[proptest]
    fn time_returns_max_by_default(d: Depth, n: u64) {
        let limits = Limits::none().with_depth(d).with_nodes(n);
        assert_eq!(limits.max_time(), Duration::MAX);
    }
}
