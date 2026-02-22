use crate::search::{Depth, Pv, Score};
use crate::util::Num;
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

const impl Info {
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

    /// The search score.
    #[inline(always)]
    pub fn score(&self) -> Score {
        self.pv.score()
    }

    /// The principal variation.
    #[inline(always)]
    pub fn pv(&self) -> &Pv {
        &self.pv
    }
}

impl const From<Pv> for Info {
    #[inline(always)]
    fn from(pv: Pv) -> Self {
        Info::new(Depth::new(0), Duration::ZERO, 0, pv)
    }
}
