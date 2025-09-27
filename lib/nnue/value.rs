use crate::chess::Flip;
use crate::util::{Bounded, Integer};
use bytemuck::Zeroable;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct ValueRepr(
    #[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <ValueRepr as Integer>::Repr,
);

unsafe impl Integer for ValueRepr {
    type Repr = i16;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 3839;
}

/// A position's static evaluation.
pub type Value = Bounded<ValueRepr>;

impl Flip for Value {
    #[inline(always)]
    fn flip(self) -> Self {
        -self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn flipping_value_returns_its_negative(v: Value) {
        assert_eq!(v.flip(), -v);
    }
}
