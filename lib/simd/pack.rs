use std::mem::{transmute, transmute_copy};
use std::simd::prelude::*;

/// Trait for [`Simd<i16, _>` ] types that implement `pack`.
pub trait Pack: SimdInt<Scalar = i16> {
    /// The output [`Simd<i8, _>` ].
    type Output: SimdUint<Scalar = u8>;

    /// Truncates `self` and `x` and interleave.
    fn pack(self, x: Self) -> Self::Output;
}

#[cfg(target_feature = "avx512bw")]
impl Pack for i16x32 {
    type Output = u8x64;

    #[inline(always)]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(255)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m512i>(self);
            let b = transmute::<Self, __m512i>(x);
            transmute::<__m512i, Self::Output>(_mm512_packus_epi16(a, b))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Pack for i16x16 {
    type Output = u8x32;

    #[inline(always)]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(255)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m256i>(self);
            let b = transmute::<Self, __m256i>(x);
            transmute::<__m256i, Self::Output>(_mm256_packus_epi16(a, b))
        }
    }
}

impl Pack for i16x8 {
    type Output = u8x16;

    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(255)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m128i>(self);
            let b = transmute::<Self, __m128i>(x);
            transmute::<__m128i, Self::Output>(_mm_packus_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(255)).all());

        unsafe {
            use std::arch::aarch64::*;
            let a = vqmovun_s16(transmute::<Self, int16x8_t>(self));
            let b = vqmovun_s16(transmute::<Self, int16x8_t>(x));
            transmute::<uint8x16_t, Self::Output>(vcombine_u8(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg(not(target_feature = "neon"))]
    fn pack(self, x: Self) -> Self::Output {
        fallback(self, x)
    }
}

#[allow(unused)]
#[inline(always)]
fn fallback(a: i16x8, b: i16x8) -> u8x16 {
    debug_assert!(b.simd_lt(Simd::splat(255)).all());
    let a = a.simd_max(Simd::splat(0)).cast::<u8>();
    let b = b.simd_max(Simd::splat(0)).cast::<u8>();
    unsafe { transmute_copy::<[u8x8; 2], u8x16>(&[a, b]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg(target_feature = "avx512bw")]
    fn for_i16x32(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x32::from_array))]
        a: i16x32,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x32::from_array))]
        b: i16x32,
    ) {
        use crate::simd::Halve;
        let [a0, a1] = a.halve();
        let [b0, b1] = b.halve();
        assert_eq!(a.pack(b).halve(), [a0.pack(b0), a1.pack(b1)]);
    }

    #[proptest]
    #[cfg(target_feature = "avx2")]
    fn for_i16x16(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        a: i16x16,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        b: i16x16,
    ) {
        use crate::simd::Halve;
        let [a0, a1] = a.halve();
        let [b0, b1] = b.halve();
        assert_eq!(a.pack(b).halve(), [a0.pack(b0), a1.pack(b1)]);
    }

    #[proptest]
    fn for_i16x8(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        a: i16x8,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        b: i16x8,
    ) {
        assert_eq!(a.pack(b), fallback(a, b));
    }
}
