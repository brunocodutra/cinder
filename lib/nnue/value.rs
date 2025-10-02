use crate::chess::Flip;
use crate::util::{Binary, Bits, Bounded, Integer};
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

impl Binary for Value {
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
    fn flipping_value_returns_its_negative(v: Value) {
        assert_eq!(v.flip(), -v);
    }

    #[proptest]
    fn decoding_encoded_value_is_an_identity(v: Value) {
        assert_eq!(Value::decode(v.encode()), v);
    }

    #[proptest]
    fn decoding_encoded_optional_value_is_an_identity(v: Option<Value>) {
        assert_eq!(Option::decode(v.encode()), v);
    }
}
