use crate::chess::{Butterfly, Move, Position};
use crate::search::{Stat, Statistics};
use crate::util::Integer;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// A linear node counter.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Nodes(usize);

unsafe impl Integer for Nodes {
    type Repr = usize;
}

impl Stat for Nodes {
    type Value = <Self as Integer>::Repr;

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
#[derive(Debug, Zeroable)]
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
    pub fn nodes(&mut self, _: &Position, m: Move) -> &mut Nodes {
        &mut self.0[m.whence() as usize][m.whither() as usize]
    }
}

impl Statistics<Move> for Attention {
    type Stat = Nodes;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.nodes(pos, m).get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.nodes(pos, m).update(delta);
    }
}
