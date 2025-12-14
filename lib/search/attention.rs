use crate::chess::{Butterfly, Move};
use crate::{search::Stat, util::Int};
use bytemuck::{Pod, Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// A linear node counter.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Zeroable, Pod)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Nodes(usize);

unsafe impl const Int for Nodes {
    type Repr = usize;
}

impl Stat for Nodes {
    type Value = <Self as Int>::Repr;

    #[inline(always)]
    fn get(&mut self) -> Self::Value {
        self.0
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        self.0 += delta;
    }
}

/// Measures the effort spent searching a root [`Move`].
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
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
    pub fn nodes(&mut self, m: Move) -> &mut Nodes {
        &mut self.0[m.whence() as usize][m.whither() as usize]
    }
}
