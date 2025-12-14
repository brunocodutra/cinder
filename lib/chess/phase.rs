use crate::util::Int;

/// The game phase.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Phase(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Phase as Int>::Repr);

impl Phase {
    pub const LEN: usize = Phase::MAX as usize + 1;
}

unsafe impl const Int for Phase {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 7;
}
