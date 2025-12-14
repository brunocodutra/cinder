use crate::util::{Assume, Binary, Bits, Bounded, Int};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Zeroable, Pod)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct DepthRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <DepthRepr as Int>::Repr);

unsafe impl const Int for DepthRepr {
    type Repr = i8;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: Self::Repr = 63;

    #[cfg(test)]
    const MAX: Self::Repr = 7;
}

/// The search depth.
pub type Depth = Bounded<DepthRepr>;

impl const Binary for Depth {
    type Bits = Bits<u8, 6>;

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
