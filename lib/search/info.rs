use crate::search::{Depth, Pv};
use derive_more::with_trait::{Constructor, Deref};
use std::hash::{Hash, Hasher};
use std::{cmp::Ordering, time::Duration};

/// Information about the search result.
#[derive(Debug, Clone, Deref, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Info {
    depth: Depth,
    time: Duration,
    nodes: u64,
    #[deref]
    pv: Pv,
}

impl Eq for Info {}

impl PartialEq for Info {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        (self.depth(), self.score()).eq(&(other.depth(), other.score()))
    }
}

impl Ord for Info {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        (self.depth(), self.score()).cmp(&(other.depth(), other.score()))
    }
}

impl PartialOrd for Info {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Info {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.depth(), self.score()).hash(state);
    }
}

impl Info {
    /// The depth searched.
    pub fn depth(&self) -> Depth {
        self.depth
    }

    /// The duration searched.
    pub fn time(&self) -> Duration {
        self.time
    }

    /// The number of nodes searched.
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// The number of nodes searched per second.
    pub fn nps(&self) -> f64 {
        self.nodes as f64 / self.time().as_secs_f64()
    }

    /// The principal variation.
    pub fn pv(&self) -> &Pv {
        &self.pv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    #[allow(clippy::nonminimal_bool, clippy::double_comparisons)]
    fn info_ordering_is_consistent(i: Info, j: Info) {
        assert_eq!(i == j, i.partial_cmp(&j) == Some(Ordering::Equal));
        assert_eq!(i < j, i.partial_cmp(&j) == Some(Ordering::Less));
        assert_eq!(i > j, i.partial_cmp(&j) == Some(Ordering::Greater));
        assert_eq!(i <= j, i < j || i == j);
        assert_eq!(i >= j, i > j || i == j);
        assert_eq!(i != j, !(i == j));
    }

    #[proptest]
    fn info_with_larger_depth_is_larger(i: Info, #[filter(#i.depth() != #j.depth())] j: Info) {
        assert_eq!(i < j, i.depth() < j.depth());
    }

    #[proptest]
    fn info_with_same_depth_but_larger_score_is_larger(
        i: Info,
        #[filter(#i.depth() == #j.depth())] j: Info,
    ) {
        assert_eq!(i < j, i.score() < j.score());
    }
}
