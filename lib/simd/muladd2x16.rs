pub use std::simd::prelude::*;

/// Trait for [`Simd<i16, _>` ] types that implement `mul_add_2x16`.
pub trait MulAdd2x16: SimdInt<Scalar = i16> {
    /// The output [`Simd<i32, _>` ].
    type Output: SimdInt<Scalar = i32>;

    /// Multiplies with the corresponding group of 2 non-negative `i16` in `x` and sums up with the corresponding `i32` in `y`.
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output;
}

impl MulAdd2x16 for i16x32 {
    type Output = i32x16;

    #[inline(always)]
    #[cfg(target_feature = "avx512vnni")]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::{arch::x86_64::*, mem::transmute};
            transmute::<__m512i, Self::Output>(_mm512_dpwssd_epi32(
                transmute::<Self::Output, __m512i>(y),
                transmute::<Self, __m512i>(x),
                transmute::<Self, __m512i>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vnni"))]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        crate::simd::Mul2x16::mul_2x16(self, x) + y
    }
}

impl MulAdd2x16 for i16x16 {
    type Output = i32x8;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::{arch::x86_64::*, mem::transmute};
            transmute::<__m256i, Self::Output>(_mm256_dpwssd_epi32(
                transmute::<Self::Output, __m256i>(y),
                transmute::<Self, __m256i>(x),
                transmute::<Self, __m256i>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        crate::simd::Mul2x16::mul_2x16(self, x) + y
    }
}

impl MulAdd2x16 for i16x8 {
    type Output = i32x4;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::{arch::x86_64::*, mem::transmute};
            transmute::<__m128i, Self::Output>(_mm_dpwssd_epi32(
                transmute::<Self::Output, __m128i>(y),
                transmute::<Self, __m128i>(x),
                transmute::<Self, __m128i>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        crate::simd::Mul2x16::mul_2x16(self, x) + y
    }
}

impl MulAdd2x16 for i16x4 {
    type Output = i32x2;

    #[inline(always)]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        crate::simd::Mul2x16::mul_2x16(self, x) + y
    }
}

impl MulAdd2x16 for i16x2 {
    type Output = i32x1;

    #[inline(always)]
    fn mul_add_2x16(self, x: Self, y: Self::Output) -> Self::Output {
        crate::simd::Mul2x16::mul_2x16(self, x) + y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::Mul2x16;
    use proptest::{array::*, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    fn for_i32x16(
        #[strategy(uniform32(-128i16..=127i16).prop_map(i16x32::from_array))] w: i16x32,
        #[strategy(uniform32(0i16..=127i16).prop_map(i16x32::from_array))] x: i16x32,
        #[strategy(uniform16(-128i32..=127i32).prop_map(i32x16::from_array))] y: i32x16,
    ) {
        assert_eq!(w.mul_add_2x16(x, y), w.mul_2x16(x) + y);
    }

    #[proptest]
    fn for_i32x8(
        #[strategy(uniform16(-128i16..=127i16).prop_map(i16x16::from_array))] w: i16x16,
        #[strategy(uniform16(0i16..=127i16).prop_map(i16x16::from_array))] x: i16x16,
        #[strategy(uniform8(-128i32..=127i32).prop_map(i32x8::from_array))] y: i32x8,
    ) {
        assert_eq!(w.mul_add_2x16(x, y), w.mul_2x16(x) + y);
    }

    #[proptest]
    fn for_i32x4(
        #[strategy(uniform8(-128i16..=127i16).prop_map(i16x8::from_array))] w: i16x8,
        #[strategy(uniform8(0i16..=127i16).prop_map(i16x8::from_array))] x: i16x8,
        #[strategy(uniform4(-128i32..=127i32).prop_map(i32x4::from_array))] y: i32x4,
    ) {
        assert_eq!(w.mul_add_2x16(x, y), w.mul_2x16(x) + y);
    }

    #[proptest]
    fn for_i32x2(
        #[strategy(uniform4(-128i16..=127i16).prop_map(i16x4::from_array))] w: i16x4,
        #[strategy(uniform4(0i16..=127i16).prop_map(i16x4::from_array))] x: i16x4,
        #[strategy(uniform2(-128i32..=127i32).prop_map(i32x2::from_array))] y: i32x2,
    ) {
        assert_eq!(w.mul_add_2x16(x, y), w.mul_2x16(x) + y);
    }

    #[proptest]
    fn for_i32x1(
        #[strategy(uniform2(-128i16..=127i16).prop_map(i16x2::from_array))] w: i16x2,
        #[strategy(uniform2(0i16..=127i16).prop_map(i16x2::from_array))] x: i16x2,
        #[strategy(uniform1(-128i32..=127i32).prop_map(i32x1::from_array))] y: i32x1,
    ) {
        assert_eq!(w.mul_add_2x16(x, y), w.mul_2x16(x) + y);
    }
}
