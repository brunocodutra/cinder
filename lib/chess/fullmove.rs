use crate::util::{Bounded, Int, Num};
use bytemuck::NoUninit;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct FullmoveRepr(
    #[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <FullmoveRepr as Num>::Repr,
);

const impl Default for FullmoveRepr {
    #[inline(always)]
    fn default() -> Self {
        Self::lower()
    }
}

const unsafe impl Num for FullmoveRepr {
    type Repr = i32;

    const MIN: Self::Repr = 1;
    const MAX: Self::Repr = i32::MAX;
}

const unsafe impl Int for FullmoveRepr {}

/// The move number, starting at 1 and incremented after every move by black.
pub type Fullmove = Bounded<FullmoveRepr>;
