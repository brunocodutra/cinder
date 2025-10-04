use std::mem::{transmute, transmute_copy};
pub use std::simd::{LaneCount, SupportedLaneCount, prelude::*};

/// Trait for [`Simd<i16, _>` ] types that implement `mul_2x16`.
pub trait Mul2x16: SimdInt<Scalar = i16> {
    /// The output [`Simd<i32, _>` ].
    type Output: SimdInt<Scalar = i32>;

    /// Multiplies with the corresponding group of 2 non-negative `i16` in `x` and sums up as `i32`.
    fn mul_2x16(self, x: Self) -> Self::Output;
}

#[allow(unused)]
#[inline(always)]
fn fallback<const M: usize, const N: usize>(
    w: Simd<i16, M>,
    x: Simd<i16, M>,
) -> <Simd<i16, M> as Mul2x16>::Output
where
    LaneCount<M>: SupportedLaneCount,
    LaneCount<N>: SupportedLaneCount,
    Simd<i16, M>: Mul2x16,
    Simd<i16, N>: Mul2x16,
{
    const { assert!(M == 2 * N) }

    unsafe {
        let w = transmute_copy::<Simd<i16, M>, [Simd<i16, N>; 2]>(&w);
        let x = transmute_copy::<Simd<i16, M>, [Simd<i16, N>; 2]>(&x);
        transmute_copy::<[<Simd<i16, N> as Mul2x16>::Output; 2], <Simd<i16, M> as Mul2x16>::Output>(
            &[w[0].mul_2x16(x[0]), w[1].mul_2x16(x[1])],
        )
    }
}

impl Mul2x16 for i16x32 {
    type Output = i32x16;

    #[inline(always)]
    #[cfg(target_feature = "avx512bw")]
    fn mul_2x16(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m512i>(self);
            let x = transmute::<Self, __m512i>(x);
            transmute::<__m512i, Self::Output>(_mm512_madd_epi16(x, w))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512bw"))]
    fn mul_2x16(self, x: Self) -> Self::Output {
        fallback::<32, 16>(self, x)
    }
}

impl Mul2x16 for i16x16 {
    type Output = i32x8;

    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    fn mul_2x16(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m256i>(self);
            let x = transmute::<Self, __m256i>(x);
            transmute::<__m256i, Self::Output>(_mm256_madd_epi16(x, w))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx2"))]
    fn mul_2x16(self, x: Self) -> Self::Output {
        fallback::<16, 8>(self, x)
    }
}

impl Mul2x16 for i16x8 {
    type Output = i32x4;

    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    fn mul_2x16(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_ge(Simd::splat(0)).all());

        unsafe {
            use std::arch::x86_64::*;
            let w = transmute::<Self, __m128i>(self);
            let x = transmute::<Self, __m128i>(x);
            transmute::<__m128i, Self::Output>(_mm_madd_epi16(x, w))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    fn mul_2x16(self, x: Self) -> Self::Output {
        unsafe {
            use std::arch::aarch64::*;

            let w = transmute::<Self, int16x8_t>(self);
            let x = transmute::<Self, int16x8_t>(x);

            let r = vmull_s16(vget_low_s16(w), vget_low_s16(x));
            let s = vmull_high_s16(w, x);

            transmute::<int32x4_t, Self::Output>(vpaddq_s32(r, s))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg(not(target_feature = "neon"))]
    fn mul_2x16(self, x: Self) -> Self::Output {
        fallback::<8, 4>(self, x)
    }
}

impl Mul2x16 for i16x4 {
    type Output = i32x2;

    #[inline(always)]
    fn mul_2x16(self, x: Self) -> Self::Output {
        fallback::<4, 2>(self, x)
    }
}

impl Mul2x16 for i16x2 {
    type Output = i32x1;

    #[inline(always)]
    fn mul_2x16(self, x: Self) -> Self::Output {
        unsafe {
            transmute::<i32, Self::Output>((self.cast::<i32>() * x.cast::<i32>()).reduce_sum())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::*, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    fn for_i32x16(
        #[strategy(uniform32(-128i16..=127i16).prop_map(i16x32::from_array))] w: i16x32,
        #[strategy(uniform32(0i16..=127i16).prop_map(i16x32::from_array))] x: i16x32,
    ) {
        assert_eq!(w.mul_2x16(x), fallback::<32, 16>(w, x));
    }

    #[proptest]
    fn for_i32x8(
        #[strategy(uniform16(-128i16..=127i16).prop_map(i16x16::from_array))] w: i16x16,
        #[strategy(uniform16(0i16..=127i16).prop_map(i16x16::from_array))] x: i16x16,
    ) {
        assert_eq!(w.mul_2x16(x), fallback::<16, 8>(w, x));
    }

    #[proptest]
    fn for_i32x4(
        #[strategy(uniform8(-128i16..=127i16).prop_map(i16x8::from_array))] w: i16x8,
        #[strategy(uniform8(0i16..=127i16).prop_map(i16x8::from_array))] x: i16x8,
    ) {
        assert_eq!(w.mul_2x16(x), fallback::<8, 4>(w, x));
    }

    #[proptest]
    fn for_i32x2(
        #[strategy(uniform4(-128i16..=127i16).prop_map(i16x4::from_array))] w: i16x4,
        #[strategy(uniform4(0i16..=127i16).prop_map(i16x4::from_array))] x: i16x4,
    ) {
        assert_eq!(w.mul_2x16(x), fallback::<4, 2>(w, x));
    }

    #[proptest]
    fn for_i32x1(
        #[strategy(uniform2(-128i16..=127i16).prop_map(i16x2::from_array))] w: i16x2,
        #[strategy(uniform2(0i16..=127i16).prop_map(i16x2::from_array))] x: i16x2,
    ) {
        assert_eq!(
            w.mul_2x16(x).reduce_sum(),
            (w.cast::<i32>() * x.cast::<i32>()).reduce_sum()
        );
    }
}
