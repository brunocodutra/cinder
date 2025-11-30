use std::{array, mem::transmute, ops::Mul, simd::prelude::*};

/// Trait for [`Simd<i8, _>` ] types that implement `mul_4x8`.
pub trait Mul4x8: SimdInt<Scalar = i8> {
    /// The output [`Simd<i32, _>` ].
    type Output: SimdInt<Scalar = i32>;

    /// Multiplies with the corresponding group of 4 non-negative `i8` in `x` and sums up as `i32`.
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output;
}

#[cfg(target_feature = "avx512bw")]
impl Mul4x8 for i8x64 {
    type Output = i32x16;

    #[inline(always)]
    #[cfg(target_feature = "avx512vnni")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vnni"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m512i>(self);
            let x = transmute::<Self::Unsigned, __m512i>(x);
            transmute::<__m512i, Self::Output>(_mm512_madd_epi16(
                _mm512_maddubs_epi16(x, w),
                _mm512_set1_epi16(1),
            ))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Mul4x8 for i8x32 {
    type Output = i32x8;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m256i>(self);
            let x = transmute::<Self::Unsigned, __m256i>(x);
            transmute::<__m256i, Self::Output>(_mm256_madd_epi16(
                _mm256_maddubs_epi16(x, w),
                _mm256_set1_epi16(1),
            ))
        }
    }
}

impl Mul4x8 for i8x16 {
    type Output = i32x4;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(target_feature = "ssse3")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m128i>(self);
            let x = transmute::<Self::Unsigned, __m128i>(x);
            transmute::<__m128i, Self::Output>(_mm_madd_epi16(
                _mm_maddubs_epi16(x, w),
                _mm_set1_epi16(1),
            ))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "dotprod")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(target_feature = "dotprod"))]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::aarch64::*;

            let w = transmute::<Self, int8x16_t>(self);
            let x = transmute::<Self::Unsigned, int8x16_t>(x);

            let r = vmull_s8(vget_low_s8(w), vget_low_s8(x));
            let s = vmull_high_s8(w, x);

            transmute::<int32x4_t, Self::Output>(vpaddq_s32(vpaddlq_s16(r), vpaddlq_s16(s)))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(not(target_feature = "ssse3"))]
    #[cfg(not(target_feature = "dotprod"))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn mul_4x8(self, x: Self::Unsigned) -> Self::Output {
        fallback(self, x)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback(w: i8x16, x: u8x16) -> i32x4 {
    debug_assert!(x.simd_lt(Simd::splat(128)).all());

    unsafe {
        let ws = transmute::<&i8x16, &[i8x4; 4]>(&w);
        let xs = transmute::<&u8x16, &[u8x4; 4]>(&x);
        Simd::from_array(array::from_fn(|i| {
            ws[i].cast::<i32>().mul(xs[i].cast::<i32>()).reduce_sum()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg(target_feature = "avx512bw")]
    fn for_i8x64(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x64::from_array))] w: i8x64,
        #[strategy(UniformArrayStrategy::new(0u8..=127u8).prop_map(u8x64::from_array))] x: u8x64,
    ) {
        use crate::simd::Halve;
        let [w0, w1] = w.halve();
        let [x0, x1] = x.halve();
        assert_eq!(w.mul_4x8(x).halve(), [w0.mul_4x8(x0), w1.mul_4x8(x1)]);
    }

    #[proptest]
    #[cfg(target_feature = "avx2")]
    fn for_i8x32(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x32::from_array))] w: i8x32,
        #[strategy(UniformArrayStrategy::new(0u8..=127u8).prop_map(u8x32::from_array))] x: u8x32,
    ) {
        use crate::simd::Halve;
        let [w0, w1] = w.halve();
        let [x0, x1] = x.halve();
        assert_eq!(w.mul_4x8(x).halve(), [w0.mul_4x8(x0), w1.mul_4x8(x1)]);
    }

    #[proptest]
    fn for_i8x16(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x16::from_array))] w: i8x16,
        #[strategy(UniformArrayStrategy::new(0u8..=127u8).prop_map(u8x16::from_array))] x: u8x16,
    ) {
        assert_eq!(w.mul_4x8(x), fallback(w, x));
    }
}
