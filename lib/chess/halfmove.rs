use crate::util::{Bounded, Int, Num};
use bytemuck::{NoUninit, Zeroable};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct HalfmoveRepr(
    #[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <HalfmoveRepr as Num>::Repr,
);

const unsafe impl Num for HalfmoveRepr {
    type Repr = i8;

    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 100;
}

const unsafe impl Int for HalfmoveRepr {}

/// The number of plies since the last capture or pawn advance.
pub type Halfmove = Bounded<HalfmoveRepr>;
