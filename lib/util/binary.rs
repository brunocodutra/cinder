use crate::util::{Assume, Bits, Int, Unsigned};

/// Trait for types that can be encoded to binary.
pub trait Binary: 'static + Sized {
    /// A fixed width collection of bits.
    type Bits: Int<Repr: Unsigned>;

    /// Encodes `Self` to its binary representation.
    fn encode(&self) -> Self::Bits;

    /// Decodes `Self` from its binary representation.
    fn decode(bits: Self::Bits) -> Self;
}

impl<T: Unsigned, const W: u32> Binary for Bits<T, W> {
    type Bits = Self;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        *self
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        bits
    }
}

impl<T: Binary<Bits: Default + Eq>> Binary for Option<T> {
    type Bits = T::Bits;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        match self {
            None => T::Bits::default(),
            Some(t) => {
                let bits = t.encode();
                (bits != T::Bits::default()).assume();
                bits
            }
        }
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        if bits == T::Bits::default() {
            None
        } else {
            Some(T::decode(bits))
        }
    }
}

macro_rules! impl_binary_for {
    ($i: ty) => {
        impl Binary for $i {
            type Bits = Bits<$i, { <$i>::BITS }>;

            #[inline(always)]
            fn encode(&self) -> Self::Bits {
                Bits::new(*self)
            }

            #[inline(always)]
            fn decode(bits: Self::Bits) -> Self {
                bits.get()
            }
        }
    };
}

impl_binary_for!(u8);
impl_binary_for!(u16);
impl_binary_for!(u32);
impl_binary_for!(u64);
impl_binary_for!(u128);
impl_binary_for!(usize);

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn encoding_bits_is_an_identity(b: Bits<u8, 6>) {
        assert_eq!(b.encode(), b);
    }

    #[proptest]
    fn decoding_bits_is_an_identity(b: Bits<u8, 6>) {
        assert_eq!(Bits::decode(b), b);
    }

    #[proptest]
    fn decoding_encoded_optional_is_an_identity(
        #[filter(#o != Some(Bits::default()))] o: Option<Bits<u8, 6>>,
    ) {
        assert_eq!(Option::decode(o.encode()), o);
    }
}
