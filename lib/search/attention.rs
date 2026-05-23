use crate::chess::{Butterfly, Move};
use crate::util::Num;
use crate::{search::Stat, util::Int};
use bytemuck::{Pod, Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// A linear node counter.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Zeroable, Pod)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Nodes(u64);

const unsafe impl Num for Nodes {
    type Repr = u64;
    const MIN: Self::Repr = u64::MIN;
    const MAX: Self::Repr = u64::MAX;
}

const unsafe impl Int for Nodes {}

impl Stat for Nodes {
    type Value = <Self as Num>::Repr;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.0
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        self.0 += delta;
    }
}

/// Measures the effort spent searching a root [`Move`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Zeroable)]
#[debug("Attention")]
pub struct Attention(Butterfly<Nodes>);

impl Default for Attention {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Attention {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn nodes(&mut self, m: Move) -> &mut Nodes {
        &mut self.0[m.whence() as usize][m.whither() as usize]
    }
}
