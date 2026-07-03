use crate::search::{Depth, Pv, Score};
use crate::util::Num;
use derive_more::with_trait::Constructor;
use std::time::Duration;

/// Information about the search result.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Info {
    time: Duration,
    depth: Depth,
    seldepth: u16,
    nodes: u64,
    tbhits: u64,
    pv: Pv,
}

impl Info {
    /// The duration searched.
    #[inline(always)]
    pub fn time(&self) -> Duration {
        self.time
    }

    /// The depth searched.
    #[inline(always)]
    pub fn depth(&self) -> Depth {
        self.depth
    }

    /// The deepest ply searched.
    #[inline(always)]
    pub fn seldepth(&self) -> u16 {
        self.seldepth
    }

    /// The number of nodes searched.
    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// The number of successful tablebase probes.
    #[inline(always)]
    pub fn tbhits(&self) -> u64 {
        self.tbhits
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

impl From<Pv> for Info {
    #[inline(always)]
    fn from(pv: Pv) -> Self {
        Info::new(Duration::ZERO, Depth::new(0), 0, 0, 0, pv)
    }
}
