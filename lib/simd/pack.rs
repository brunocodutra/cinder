use std::mem::{transmute, transmute_copy};
pub use std::simd::{LaneCount, SupportedLaneCount, prelude::*};

/// Trait for [`Simd<i16, _>` ] types that implement `pack`.
pub trait Pack: SimdInt<Scalar = i16> {
    /// The output [`Simd<i8, _>` ].
    type Output: SimdInt<Scalar = i8>;

    /// Truncates `self` and `x` and interleave.
    fn pack(self, x: Self) -> Self::Output;
}

#[allow(unused)]
#[inline(always)]
fn fallback<const M: usize, const N: usize>(
    a: Simd<i16, M>,
    b: Simd<i16, M>,
) -> <Simd<i16, M> as Pack>::Output
where
    LaneCount<M>: SupportedLaneCount,
    LaneCount<N>: SupportedLaneCount,
    Simd<i16, M>: Pack,
    Simd<i16, N>: Pack,
{
    const { assert!(M == 2 * N) }

    unsafe {
        let a = transmute_copy::<Simd<i16, M>, [Simd<i16, N>; 2]>(&a);
        let b = transmute_copy::<Simd<i16, M>, [Simd<i16, N>; 2]>(&b);
        transmute_copy::<[<Simd<i16, N> as Pack>::Output; 2], <Simd<i16, M> as Pack>::Output>(&[
            a[0].pack(b[0]),
            a[1].pack(b[1]),
        ])
    }
}

impl Pack for i16x32 {
    type Output = i8x64;

    #[inline(always)]
    #[cfg(target_feature = "avx512bw")]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m512i>(self);
            let b = transmute::<Self, __m512i>(x);
            transmute::<__m512i, Self::Output>(_mm512_packus_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512bw"))]
    fn pack(self, x: Self) -> Self::Output {
        fallback::<32, 16>(self, x)
    }
}

impl Pack for i16x16 {
    type Output = i8x32;

    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::x86_64::*;
            let a = transmute::<Self, __m256i>(self);
            let b = transmute::<Self, __m256i>(x);
            transmute::<__m256i, Self::Output>(_mm256_packus_epi16(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx2"))]
    fn pack(self, x: Self) -> Self::Output {
        fallback::<16, 8>(self, x)
    }
}

impl Pack for i16x8 {
    type Output = i8x16;

    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

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
        debug_assert!(x.simd_lt(Simd::splat(128)).all());

        unsafe {
            use std::arch::aarch64::*;
            let a = vqmovun_s16(transmute::<Self, int8x16_t>(self));
            let b = vqmovun_s16(transmute::<Self, int8x16_t>(x));
            transmute::<int8x16_t, Self::Output>(vcombine_u8(a, b))
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg(not(target_feature = "neon"))]
    fn pack(self, x: Self) -> Self::Output {
        debug_assert!(x.simd_lt(Simd::splat(128)).all());
        let a = self.max(Simd::splat(0)).cast::<i8>();
        let b = x.max(Simd::splat(0)).cast::<i8>();
        unsafe { transmute::<[i8x8; 2], i8x16>([a, b]) }
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
        assert_eq!(a.pack(b), fallback::<32, 16>(a, b));
    }

    #[proptest]
    fn for_i16x16(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        a: i16x16,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x16::from_array))]
        b: i16x16,
    ) {
        assert_eq!(a.pack(b), fallback::<16, 8>(a, b));
    }
}
