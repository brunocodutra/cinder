use crate::util::{Assume, Binary, Bits, Bounded, Int, Num};
use bytemuck::{NoUninit, Zeroable};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct DepthRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <DepthRepr as Num>::Repr);

const unsafe impl Num for DepthRepr {
    type Repr = i8;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: Self::Repr = 127;

    #[cfg(test)]
    const MAX: Self::Repr = 3;
}

const unsafe impl Int for DepthRepr {}

/// The search depth.
pub type Depth = Bounded<DepthRepr>;

impl Binary for Depth {
    type Bits = Bits<u8, 7>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        self.convert().assume()
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        bits.convert().assume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn decoding_encoded_depth_is_an_identity(d: Depth) {
        assert_eq!(Depth::decode(d.encode()), d);
    }
}
