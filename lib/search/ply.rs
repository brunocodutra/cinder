use crate::util::{Bounded, Integer};

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PlyRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

unsafe impl Integer for PlyRepr {
    type Repr = i8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 95;
}

/// The number of half-moves played.
pub type Ply = Bounded<PlyRepr>;
