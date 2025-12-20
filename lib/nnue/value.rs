use crate::chess::Flip;
use crate::util::{Binary, Bits, Bounded, Int};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Copy, Hash, Zeroable, Pod)]
#[derive_const(Default, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct ValueRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <ValueRepr as Int>::Repr);

unsafe impl const Int for ValueRepr {
    type Repr = i16;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 3839;
}

/// A position's static evaluation.
pub type Value = Bounded<ValueRepr>;

impl const Flip for Value {
    #[inline(always)]
    fn flip(self) -> Self {
        -self
    }
}

impl const Binary for Value {
    type Bits = Bits<u16, 13>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        Bits::new((self.get() - Self::MIN + 1).cast())
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Value::new(bits.cast::<i16>() + Self::MIN - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn flipping_value_returns_its_negative(v: Value) {
        assert_eq!(v.flip(), -v);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_value_is_an_identity(v: Value) {
        assert_eq!(Value::decode(v.encode()), v);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_optional_value_is_an_identity(v: Option<Value>) {
        assert_eq!(Option::decode(v.encode()), v);
    }
}
