use crate::util::{Bounded, Int};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Copy, Hash, Zeroable, Pod)]
#[derive_const(Default, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PlyRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PlyRepr as Int>::Repr);

unsafe impl const Int for PlyRepr {
    type Repr = i8;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: Self::Repr = 127;

    #[cfg(test)]
    const MAX: Self::Repr = 15;
}

/// The number of half-moves played.
pub type Ply = Bounded<PlyRepr>;
