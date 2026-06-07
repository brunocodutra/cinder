use crate::simd::Halve;
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
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn pack(self, x: Self) -> Self::Output {
        unsafe {
            use std::arch::x86_64::*;
            _mm512_packus_epi16(self.into(), x.into()).into()
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Pack for i16x16 {
    type Output = u8x32;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn pack(self, x: Self) -> Self::Output {
        unsafe {
            use std::arch::x86_64::*;
            _mm256_packus_epi16(self.into(), x.into()).into()
        }
    }
}

impl Pack for i16x8 {
    type Output = u8x16;

    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn pack(self, x: Self) -> Self::Output {
        unsafe {
            use std::arch::x86_64::*;
            _mm_packus_epi16(self.into(), x.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn pack(self, x: Self) -> Self::Output {
        unsafe {
            use std::arch::aarch64::*;
            let a = vqmovun_s16(self.into()).into();
            let b = vqmovun_s16(x.into()).into();
            vcombine_u8(a, b).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn pack(self, x: Self) -> Self::Output {
        fallback(self, x)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback(a: i16x8, b: i16x8) -> u8x16 {
    let a = a.simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>();
    let b = b.simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>();
    Halve::merge([a, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
    fn for_i16x8(
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        a: i16x8,
        #[strategy(UniformArrayStrategy::new(-128i16..=127i16).prop_map(i16x8::from_array))]
        b: i16x8,
    ) {
        assert_eq!(a.pack(b), fallback(a, b));
    }
}
