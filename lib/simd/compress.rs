use crate::util::{Unsigned, ones, zero};
use std::simd::{SimdElement, prelude::*};

/// Trait for [`Simd<_, _>` ] types that can be compressed given indices in a bitmask.
pub trait Compress {
    type Bitmask: Unsigned;

    /// Compacts elements from `self` that match `indices`.
    fn compress(self, indices: Self::Bitmask) -> Self;
}

impl Compress for u8x64 {
    type Bitmask = u64;

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm512_maskz_compress_epi8(indices, self.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        fallback(self, indices)
    }
}

impl Compress for u16x32 {
    type Bitmask = u32;

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm512_maskz_compress_epi16(indices, self.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        fallback(self, indices)
    }
}

impl Compress for u16x16 {
    type Bitmask = u16;

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm256_maskz_compress_epi16(indices, self.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        fallback(self, indices)
    }
}

impl Compress for u16x8 {
    type Bitmask = u8;

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm_maskz_compress_epi16(indices, self.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress(self, indices: Self::Bitmask) -> Self {
        fallback(self, indices)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback<T: Unsigned + SimdElement, U: Unsigned, const N: usize>(
    w: Simd<T, N>,
    mut indices: U,
) -> Simd<T, N> {
    let mut compressed = [zero(); N];

    let mut cursor = 0;
    while indices != zero() {
        compressed[cursor] = w[indices.trailing_zeros() as usize];
        indices &= indices.wrapping_sub(ones(1));
        cursor += 1;
    }

    Simd::from_array(compressed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x64(
        #[strategy(UniformArrayStrategy::new(0u8..).prop_map(u8x64::from_array))] w: u8x64,
        m: u64,
    ) {
        assert_eq!(w.compress(m), fallback(w, m));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x32(
        #[strategy(UniformArrayStrategy::new(0u16..).prop_map(u16x32::from_array))] w: u16x32,
        m: u32,
    ) {
        assert_eq!(w.compress(m), fallback(w, m));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x16(
        #[strategy(UniformArrayStrategy::new(0u16..).prop_map(u16x16::from_array))] w: u16x16,
        m: u16,
    ) {
        assert_eq!(w.compress(m), fallback(w, m));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x8(
        #[strategy(UniformArrayStrategy::new(0u16..).prop_map(u16x8::from_array))] w: u16x8,
        m: u8,
    ) {
        assert_eq!(w.compress(m), fallback(w, m));
    }
}
