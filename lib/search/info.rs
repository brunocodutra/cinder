use crate::search::{Depth, Pv, Score};
use crate::util::Int;
use derive_more::with_trait::Constructor;
use std::time::Duration;

/// Information about the search result.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Info {
    depth: Depth,
    time: Duration,
    nodes: u64,
    pv: Pv,
}

impl Info {
    /// The depth searched.
    #[inline(always)]
    pub const fn depth(&self) -> Depth {
        self.depth
    }

    /// The duration searched.
    #[inline(always)]
    pub const fn time(&self) -> Duration {
        self.time
    }

    /// The number of nodes searched.
    #[inline(always)]
    pub const fn nodes(&self) -> u64 {
        self.nodes
    }

    /// The number of nodes searched per second.
    #[inline(always)]
    pub const fn nps(&self) -> f64 {
        self.nodes as f64 / self.time().as_secs_f64().max(1E-6)
    }

    /// The search score.
    #[inline(always)]
    pub const fn score(&self) -> Score {
        self.pv.score()
    }

    /// The principal variation.
    #[inline(always)]
    pub const fn pv(&self) -> &Pv {
        &self.pv
    }
}

impl const From<Pv> for Info {
    #[inline(always)]
    fn from(pv: Pv) -> Self {
        Info::new(Depth::new(0), Duration::ZERO, 0, pv)
    }
}
