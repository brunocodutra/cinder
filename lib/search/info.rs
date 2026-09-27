use crate::search::{Depth, Pv, Score};
use crate::util::Num;
use std::time::Duration;

/// Information about the search result.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Info {
    depth: Depth,
    seldepth: u16,
    time: Duration,
    nodes: u64,
    tbhits: u64,
    hashfull: f32,
    pv: Pv,
}

impl Info {
    /// The duration searched.
    #[inline(always)]
    pub fn new<const N: usize>(
        depth: Depth,
        seldepth: u16,
        time: Duration,
        nodes: u64,
        tbhits: u64,
        hashfull: f32,
        pv: Pv<N>,
    ) -> Self {
        Self {
            depth,
            seldepth,
            time,
            nodes,
            tbhits,
            hashfull,
            pv: pv.truncate(),
        }
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

    /// The number of successful tablebase probes.
    #[inline(always)]
    pub fn tbhits(&self) -> u64 {
        self.tbhits
    }

    /// The fraction of transposition table slots in use.
    #[inline(always)]
    pub fn hashfull(&self) -> f32 {
        self.hashfull
    }

    /// The search score.
    #[inline(always)]
    pub fn score(&self) -> Score {
        self.pv.score()
    }

    /// The principal variation.
    #[inline(always)]
    pub fn pv(&self) -> Pv {
        self.pv
    }
}

impl From<Pv> for Info {
    #[inline(always)]
    fn from(pv: Pv) -> Self {
        Info::new(Depth::new(0), 0, Duration::ZERO, 0, 0, 0.0, pv)
    }
}
