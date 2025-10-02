use std::mem::{transmute, transmute_copy};
pub use std::simd::{LaneCount, SupportedLaneCount, prelude::*};

/// Trait for [`Simd<i8, _>` ] types that implement `mul_4x8`.
pub trait Mul4x8: SimdInt<Scalar = i8> {
    /// The output [`Simd<i32, _>` ].
    type Output: SimdInt<Scalar = i32>;

    /// Multiplies with the corresponding group of 4 non-negative `i8` in `x` and sums up as `i32`.
    fn mul_4x8(self, x: Self) -> Self::Output;
}

#[allow(unused)]
#[inline(always)]
fn fallback<const M: usize, const N: usize>(
    w: Simd<i8, M>,
    x: Simd<i8, M>,
) -> <Simd<i8, M> as Mul4x8>::Output
where
    LaneCount<M>: SupportedLaneCount,
    LaneCount<N>: SupportedLaneCount,
    Simd<i8, M>: Mul4x8,
    Simd<i8, N>: Mul4x8,
{
    const { assert!(M == 2 * N) }

    unsafe {
        let w = transmute_copy::<Simd<i8, M>, [Simd<i8, N>; 2]>(&w);
        let x = transmute_copy::<Simd<i8, M>, [Simd<i8, N>; 2]>(&x);
        transmute_copy::<[<Simd<i8, N> as Mul4x8>::Output; 2], <Simd<i8, M> as Mul4x8>::Output>(&[
            w[0].mul_4x8(x[0]),
            w[1].mul_4x8(x[1]),
        ])
    }
}

impl Mul4x8 for i8x64 {
    type Output = i32x16;

    #[inline(always)]
    #[cfg(target_feature = "avx512vnni")]
    fn mul_4x8(self, x: Self) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vnni"))]
    #[cfg(target_feature = "avx512bw")]
    fn mul_4x8(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m512i>(self);
            let x = transmute::<Self, __m512i>(x);
            transmute::<__m512i, Self::Output>(_mm512_madd_epi16(
                _mm512_maddubs_epi16(x, w),
                _mm512_set1_epi16(1),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vnni"))]
    #[cfg(not(target_feature = "avx512bw"))]
    fn mul_4x8(self, x: Self) -> Self::Output {
        fallback::<64, 32>(self, x)
    }
}

impl Mul4x8 for i8x32 {
    type Output = i32x8;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    fn mul_4x8(self, x: Self) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(target_feature = "avx2")]
    fn mul_4x8(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m256i>(self);
            let x = transmute::<Self, __m256i>(x);
            transmute::<__m256i, Self::Output>(_mm256_madd_epi16(
                _mm256_maddubs_epi16(x, w),
                _mm256_set1_epi16(1),
            ))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(not(target_feature = "avx2"))]
    fn mul_4x8(self, x: Self) -> Self::Output {
        fallback::<32, 16>(self, x)
    }
}

impl Mul4x8 for i8x16 {
    type Output = i32x4;

    #[inline(always)]
    #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vl"))]
    fn mul_4x8(self, x: Self) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(target_feature = "ssse3")]
    fn mul_4x8(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m128i>(self);
            let x = transmute::<Self, __m128i>(x);
            transmute::<__m128i, Self::Output>(_mm_madd_epi16(
                _mm_maddubs_epi16(x, w),
                _mm_set1_epi16(1),
            ))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "dotprod")]
    fn mul_4x8(self, x: Self) -> Self::Output {
        crate::simd::MulAdd4x8::mul_add_4x8(self, x, Simd::splat(0))
    }

    #[inline(always)]
    #[cfg(not(target_feature = "dotprod"))]
    #[cfg(target_feature = "neon")]
    fn mul_4x8(self, x: Self) -> Self::Output {
        unsafe {
            use std::arch::aarch64::*;

            let w = transmute::<Self, int8x16_t>(self);
            let x = transmute::<Self, int8x16_t>(x);

            let r = vmull_s8(vget_low_s8(w), vget_low_s8(x));
            let s = vmull_high_s8(w, x);

            transmute::<int32x4_t, Self::Output>(vaddq_s32(vpaddlq_s16(r), vpaddlq_s16(s)))
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vl")))]
    #[cfg(not(target_feature = "ssse3"))]
    #[cfg(not(target_feature = "dotprod"))]
    #[cfg(not(target_feature = "neon"))]
    fn mul_4x8(self, x: Self) -> Self::Output {
        fallback::<16, 8>(self, x)
    }
}

impl Mul4x8 for i8x8 {
    type Output = i32x2;

    #[inline(always)]
    fn mul_4x8(self, x: Self) -> Self::Output {
        fallback::<8, 4>(self, x)
    }
}

impl Mul4x8 for i8x4 {
    type Output = i32x1;

    #[inline(always)]
    fn mul_4x8(self, x: Self) -> Self::Output {
        unsafe {
            transmute::<i32, Self::Output>((self.cast::<i32>() * x.cast::<i32>()).reduce_sum())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    fn for_i8x64(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x64::from_array))] w: i8x64,
        #[strategy(UniformArrayStrategy::new(0i8..=127i8).prop_map(i8x64::from_array))] x: i8x64,
    ) {
        assert_eq!(w.mul_4x8(x), fallback::<64, 32>(w, x));
    }

    #[proptest]
    fn for_i8x32(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x32::from_array))] w: i8x32,
        #[strategy(UniformArrayStrategy::new(0i8..=127i8).prop_map(i8x32::from_array))] x: i8x32,
    ) {
        assert_eq!(w.mul_4x8(x), fallback::<32, 16>(w, x));
    }

    #[proptest]
    fn for_i8x16(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x16::from_array))] w: i8x16,
        #[strategy(UniformArrayStrategy::new(0i8..=127i8).prop_map(i8x16::from_array))] x: i8x16,
    ) {
        assert_eq!(w.mul_4x8(x), fallback::<16, 8>(w, x));
    }

    #[proptest]
    fn for_i8x8(
        #[strategy(UniformArrayStrategy::new(-128i8..=127i8).prop_map(i8x8::from_array))] w: i8x8,
        #[strategy(UniformArrayStrategy::new(0i8..=127i8).prop_map(i8x8::from_array))] x: i8x8,
    ) {
        assert_eq!(w.mul_4x8(x), fallback::<8, 4>(w, x));
    }
}
