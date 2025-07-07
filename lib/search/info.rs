use crate::search::{Depth, Pv};
use derive_more::with_trait::{Constructor, Deref};
use std::time::Duration;

/// Information about the search result.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Info {
    depth: Depth,
    time: Duration,
    nodes: u64,
    #[deref]
    pv: Pv,
}

impl Info {
    /// The depth searched.
    #[inline(always)]
    pub fn depth(&self) -> Depth {
        self.depth
    }

    /// The duration searched.
    #[inline(always)]
    pub fn time(&self) -> Duration {
        self.time
    }

    /// The number of nodes searched.
    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// The number of nodes searched per second.
    #[inline(always)]
    pub fn nps(&self) -> f64 {
        self.nodes as f64 / self.time().max(Duration::from_nanos(1)).as_secs_f64()
    }

    /// The principal variation.
    #[inline(always)]
    pub fn pv(&self) -> &Pv {
        &self.pv
    }
}
