use crate::util::{Int, Num};

/// The game phase.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Phase(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Phase as Num>::Repr);

const impl Phase {
    pub const LEN: usize = Phase::MAX as usize + 1;
}

const unsafe impl Num for Phase {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 7;
}

const unsafe impl Int for Phase {}
