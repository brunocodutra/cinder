use crate::util::Integer;

/// The game phase.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Phase(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

unsafe impl Integer for Phase {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 7;
}
