use crate::util::{Bounded, Int, Num};
use bytemuck::{NoUninit, Zeroable};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PlyRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PlyRepr as Num>::Repr);

const unsafe impl Num for PlyRepr {
    type Repr = i16;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: Self::Repr = 159;

    #[cfg(test)]
    const MAX: Self::Repr = 7;
}

const unsafe impl Int for PlyRepr {}

/// The number of half-moves played.
pub type Ply = Bounded<PlyRepr>;
