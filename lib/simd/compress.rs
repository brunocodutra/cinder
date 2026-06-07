use crate::util::{Unsigned, ones};
use bytemuck::{Zeroable, zeroed};
use std::simd::{SimdElement, prelude::*};

/// Trait for [`Simd<_, _>` ] types that can be compressed given indices in a bitmask.
pub trait Compress: SimdUint {
    type Bitmask: Unsigned;

    /// Compacts elements from `self` that match `indices`.
    fn compress(self, indices: Self::Bitmask) -> Self;

    /// Compacts elements from `self` that match `indices` and writes to `slice`.
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]);
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
        fallback_compress(self, indices)
    }

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        use crate::util::Assume;
        (slice.len() >= Self::LEN).assume();
        self.compress(indices).copy_to_slice(slice);
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        fallback_compress_store(self, indices, slice);
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
        fallback_compress(self, indices)
    }

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        use crate::util::Assume;
        (slice.len() >= Self::LEN).assume();
        self.compress(indices).copy_to_slice(slice);
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        fallback_compress_store(self, indices, slice);
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
        fallback_compress(self, indices)
    }

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        use crate::util::Assume;
        (slice.len() >= Self::LEN).assume();
        self.compress(indices).copy_to_slice(slice);
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        fallback_compress_store(self, indices, slice);
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
        fallback_compress(self, indices)
    }

    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        use crate::util::Assume;
        (slice.len() >= Self::LEN).assume();
        self.compress(indices).copy_to_slice(slice);
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn compress_store(self, indices: Self::Bitmask, slice: &mut [Self::Scalar]) {
        fallback_compress_store(self, indices, slice);
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback_compress<T: SimdElement + Zeroable, U: Unsigned, const N: usize>(
    w: Simd<T, N>,
    indices: U,
) -> Simd<T, N> {
    let mut compressed: [T; N] = zeroed();
    fallback_compress_store(w, indices, &mut compressed);
    Simd::from_array(compressed)
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback_compress_store<T: SimdElement, U: Unsigned, const N: usize>(
    w: Simd<T, N>,
    mut indices: U,
    slice: &mut [T],
) {
    let src = w.to_array();
    let dst = slice.as_mut_ptr();

    let mut cursor = 0;
    while indices != zeroed() {
        let bit = indices.trailing_zeros() as usize;
        unsafe { dst.add(cursor).write(*src.as_ptr().add(bit)) };
        indices &= indices.wrapping_sub(ones(1));
        cursor += 1;
    }
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
        assert_eq!(w.compress(m), fallback_compress(w, m));

        let mut a = [0; 64];
        w.compress_store(m, &mut a);

        let mut b = [0; 64];
        fallback_compress_store(w, m, &mut b);

        assert_eq!(a, b);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x32(
        #[strategy(UniformArrayStrategy::new(0u16..).prop_map(u16x32::from_array))] w: u16x32,
        m: u32,
    ) {
        assert_eq!(w.compress(m), fallback_compress(w, m));

        let mut a = [0; 32];
        w.compress_store(m, &mut a);

        let mut b = [0; 32];
        fallback_compress_store(w, m, &mut b);

        assert_eq!(a, b);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x16(
        #[strategy(UniformArrayStrategy::new(0u16..).prop_map(u16x16::from_array))] w: u16x16,
        m: u16,
    ) {
        assert_eq!(w.compress(m), fallback_compress(w, m));

        let mut a = [0; 16];
        w.compress_store(m, &mut a);

        let mut b = [0; 16];
        fallback_compress_store(w, m, &mut b);

        assert_eq!(a, b);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x8(
        #[strategy(UniformArrayStrategy::new(0u16..).prop_map(u16x8::from_array))] w: u16x8,
        m: u8,
    ) {
        assert_eq!(w.compress(m), fallback_compress(w, m));

        let mut a = [0; 8];
        w.compress_store(m, &mut a);

        let mut b = [0; 8];
        fallback_compress_store(w, m, &mut b);

        assert_eq!(a, b);
    }
}
