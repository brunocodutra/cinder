use std::ops::{Mul, Shl, Shr};
use std::{mem::transmute, simd::prelude::*};

/// Trait for [`Simd<i16, _>` ] types that implement `mul_high`.
pub trait MulHigh: SimdInt<Scalar = i16> {
    /// Multiplies with the corresponding `i16` in `x` discarding the lower `B` bits.
    fn mul_high<const B: usize>(self, x: Self) -> Self;
}

#[cfg(target_feature = "avx512bw")]
impl MulHigh for i16x32 {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(8 < B && B <= 16) };

        debug_assert!(x.simd_lt(Simd::splat(256)).all());
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m512i>(self);
            let b = transmute::<Self, __m512i>(x.shl(16 - B as i16));
            transmute::<__m512i, Self>(_mm512_mulhi_epi16(a, b))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl MulHigh for i16x16 {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(8 < B && B <= 16) };

        debug_assert!(x.simd_lt(Simd::splat(256)).all());
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m256i>(self);
            let b = transmute::<Self, __m256i>(x.shl(16 - B as i16));
            transmute::<__m256i, Self>(_mm256_mulhi_epi16(a, b))
        }
    }
}

impl MulHigh for i16x8 {
    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(8 < B && B <= 16) };

        debug_assert!(x.simd_lt(Simd::splat(256)).all());
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m128i>(self);
            let b = transmute::<Self, __m128i>(x.shl(16 - B as i16));
            transmute::<__m128i, Self>(_mm_mulhi_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(8 < B && B <= 16) };

        debug_assert!(x.simd_lt(Simd::splat(256)).all());
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::aarch64::*;
            let a = transmute::<Self, int16x8_t>(self);
            let b = transmute::<Self, int16x8_t>(x.shl(16 - 1 - B as i16));
            transmute::<int16x8_t, Self>(vqdmulhq_s16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        fallback::<B>(self, x)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback<const B: usize>(a: i16x8, b: i16x8) -> i16x8 {
    const { assert!(8 < B && B <= 16) };
    debug_assert!(b.simd_lt(Simd::splat(256)).all());
    debug_assert!(b.simd_ge(Simd::splat(0)).all());
    let b = b.shl(16 - B as i16).cast::<i32>();
    a.cast::<i32>().mul(b).shr(16).cast::<i16>()
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
        #[strategy(UniformArrayStrategy::new(0i16..=255i16).prop_map(i16x32::from_array))]
        b: i16x32,
    ) {
        use crate::simd::Halve;
        let [a0, a1] = a.halve();
        let [b0, b1] = b.halve();

        assert_eq!(
            a.mul_high::<9>(b).halve(),
            [a0.mul_high::<9>(b0), a1.mul_high::<9>(b1)]
        );
    }

    #[proptest]
    #[cfg(target_feature = "avx2")]
    fn for_i16x16(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        a: i16x16,
        #[strategy(UniformArrayStrategy::new(0i16..=255i16).prop_map(i16x16::from_array))]
        b: i16x16,
    ) {
        use crate::simd::Halve;
        let [a0, a1] = a.halve();
        let [b0, b1] = b.halve();

        assert_eq!(
            a.mul_high::<9>(b).halve(),
            [a0.mul_high::<9>(b0), a1.mul_high::<9>(b1)]
        );
    }

    #[proptest]
    fn for_i16x8(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        a: i16x8,
        #[strategy(UniformArrayStrategy::new(0i16..=255i16).prop_map(i16x8::from_array))] b: i16x8,
    ) {
        assert_eq!(a.mul_high::<9>(b), fallback::<9>(a, b));
    }
}
