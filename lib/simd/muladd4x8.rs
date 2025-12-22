use std::simd::prelude::*;

/// Trait for [`Simd<i8, _>` ] types that implement `mul_add_4x8`.
pub trait MulAdd4x8: SimdInt<Scalar = i8> {
    /// The output [`Simd<i32, _>` ].
    type Output: SimdInt<Scalar = i32>;

    /// Multiplies with the corresponding group of 4 non-negative `i8` in `x` and sums up with the corresponding `i32` in `y`.
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output;
}

#[cfg(target_feature = "avx512bw")]
impl MulAdd4x8 for i8x64 {
    type Output = i32x16;

    #[inline(always)]
    #[cfg(target_feature = "avx512vnni")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::{arch::x86_64::*, mem::transmute};
            transmute::<__m512i, Self::Output>(_mm512_dpbusd_epi32(
                transmute::<Self::Output, __m512i>(y),
                transmute::<Self::Unsigned, __m512i>(x),
                transmute::<Self, __m512i>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vnni"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    #[expect(clippy::absolute_paths)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        crate::simd::Mul4x8::mul_4x8(self, x) + y
    }
}

#[cfg(target_feature = "avx2")]
impl MulAdd4x8 for i8x32 {
    type Output = i32x8;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::{arch::x86_64::*, mem::transmute};
            transmute::<__m256i, Self::Output>(_mm256_dpbusd_epi32(
                transmute::<Self::Output, __m256i>(y),
                transmute::<Self::Unsigned, __m256i>(x),
                transmute::<Self, __m256i>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    #[expect(clippy::absolute_paths)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        crate::simd::Mul4x8::mul_4x8(self, x) + y
    }
}

impl MulAdd4x8 for i8x16 {
    type Output = i32x4;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::{arch::x86_64::*, mem::transmute};
            transmute::<__m128i, Self::Output>(_mm_dpbusd_epi32(
                transmute::<Self::Output, __m128i>(y),
                transmute::<Self::Unsigned, __m128i>(x),
                transmute::<Self, __m128i>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "dotprod")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::{arch::aarch64::*, mem::transmute};
            transmute::<int32x4_t, Self::Output>(vdotq_s32(
                transmute::<Self::Output, int32x4_t>(y),
                transmute::<Self::Unsigned, int8x16_t>(x),
                transmute::<Self, int8x16_t>(self),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(not(target_feature = "dotprod"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    #[expect(clippy::absolute_paths)]
    fn mul_add_4x8(self, x: Self::Unsigned, y: Self::Output) -> Self::Output {
        crate::simd::Mul4x8::mul_4x8(self, x) + y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::Mul4x8;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    #[cfg(target_feature = "avx512bw")]
    fn for_i8x64(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x64::from_array))] w: i8x64,
        #[strategy(UniformArrayStrategy::new(0u8..=127u8).prop_map(u8x64::from_array))] x: u8x64,
        #[strategy(UniformArrayStrategy::new(-128i32..=127i32).prop_map(i32x16::from_array))]
        y: i32x16,
    ) {
        assert_eq!(w.mul_add_4x8(x, y), w.mul_4x8(x) + y);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    #[cfg(target_feature = "avx2")]
    fn for_i8x32(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x32::from_array))] w: i8x32,
        #[strategy(UniformArrayStrategy::new(0u8..=127u8).prop_map(u8x32::from_array))] x: u8x32,
        #[strategy(UniformArrayStrategy::new(-128i32..=127i32).prop_map(i32x8::from_array))]
        y: i32x8,
    ) {
        assert_eq!(w.mul_add_4x8(x, y), w.mul_4x8(x) + y);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_i8x16(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x16::from_array))] w: i8x16,
        #[strategy(UniformArrayStrategy::new(0u8..=127u8).prop_map(u8x16::from_array))] x: u8x16,
        #[strategy(UniformArrayStrategy::new(-128i32..=127i32).prop_map(i32x4::from_array))]
        y: i32x4,
    ) {
        assert_eq!(w.mul_add_4x8(x, y), w.mul_4x8(x) + y);
    }
}
