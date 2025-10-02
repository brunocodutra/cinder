use std::mem::{transmute, transmute_copy};
use std::ops::{Mul, Shl, Shr};
pub use std::simd::{LaneCount, SupportedLaneCount, prelude::*};

/// Trait for [`Simd<i16, _>` ] types that implement `mul_high`.
pub trait MulHigh: SimdInt<Scalar = i16> {
    /// Multiplies with the corresponding `i16` in `x` discarding the lower `B` bits.
    fn mul_high<const B: usize>(self, x: Self) -> Self;
}

#[allow(unused)]
#[inline(always)]
fn fallback<const B: usize, const M: usize, const N: usize>(
    a: Simd<i16, M>,
    b: Simd<i16, M>,
) -> Simd<i16, M>
where
    LaneCount<M>: SupportedLaneCount,
    LaneCount<N>: SupportedLaneCount,
    Simd<i16, M>: MulHigh,
    Simd<i16, N>: MulHigh,
{
    const { assert!(M == 2 * N) }

    unsafe {
        let a = transmute_copy::<Simd<i16, M>, [Simd<i16, N>; 2]>(&a);
        let b = transmute_copy::<Simd<i16, M>, [Simd<i16, N>; 2]>(&b);
        transmute_copy::<[Simd<i16, N>; 2], Simd<i16, M>>(&[
            a[0].mul_high::<B>(b[0]),
            a[1].mul_high::<B>(b[1]),
        ])
    }
}

impl MulHigh for i16x32 {
    #[inline(always)]
    #[cfg(target_feature = "avx512bw")]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(1 <= B && B <= 16) };

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m512i>(self);
            let b = transmute::<Self, __m512i>(x.shl(16 - B as i16));
            transmute::<__m512i, Self>(_mm512_mulhi_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512bw"))]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        fallback::<B, 32, 16>(self, x)
    }
}

impl MulHigh for i16x16 {
    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(1 <= B && B <= 16) };

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m256i>(self);
            let b = transmute::<Self, __m256i>(x.shl(16 - B as i16));
            transmute::<__m256i, Self>(_mm256_mulhi_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx2"))]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        fallback::<B, 16, 8>(self, x)
    }
}

impl MulHigh for i16x8 {
    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        const { assert!(1 <= B && B <= 16) };

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m128i>(self);
            let b = transmute::<Self, __m128i>(x.shl(16 - B as i16));
            transmute::<__m128i, Self>(_mm_mulhi_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = transmute::<Self, int8x16_t>(self);
            let b = transmute::<Self, int8x16_t>(x.shl(16 - 1 - B as i16));
            transmute::<int8x16_t, Self>(vqdmulhq_s16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg(not(target_feature = "neon"))]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        fallback::<B, 8, 4>(self, x)
    }
}

impl MulHigh for i16x4 {
    #[inline(always)]
    fn mul_high<const B: usize>(self, x: Self) -> Self {
        let x = x.shl(16 - B as i16).cast::<i32>();
        self.cast::<i32>().mul(x).shr(16).cast::<i16>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    fn for_i16x32(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x32::from_array))]
        a: i16x32,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x32::from_array))]
        b: i16x32,
    ) {
        assert_eq!(a.mul_high::<5>(b), fallback::<5, 32, 16>(a, b));
    }

    #[proptest]
    fn for_i16x16(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        a: i16x16,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        b: i16x16,
    ) {
        assert_eq!(a.mul_high::<5>(b), fallback::<5, 16, 8>(a, b));
    }

    #[proptest]
    fn for_i16x8(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        a: i16x8,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        b: i16x8,
    ) {
        assert_eq!(a.mul_high::<5>(b), fallback::<5, 8, 4>(a, b));
    }
}
