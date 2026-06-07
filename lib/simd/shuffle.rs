use std::simd::prelude::*;

/// Trait for [`Simd<_, _>` ] types that can shuffle within lanes.
pub trait Shuffle {
    /// Shuffles within lanes given `indices`.
    fn shuffle(self, indices: Self) -> Self;
}

impl Shuffle for u8x64 {
    #[inline(always)]
    #[cfg(target_feature = "avx512bw")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm512_shuffle_epi8(self.into(), indices.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512bw"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        use crate::simd::Halve;
        let [x0, x1] = self.halve();
        let [i0, i1] = indices.halve();
        Halve::merge([x0.shuffle(i0), x1.shuffle(i1)])
    }
}

impl Shuffle for u8x32 {
    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm256_shuffle_epi8(self.into(), indices.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        use crate::simd::Halve;
        let [x0, x1] = self.halve();
        let [i0, i1] = indices.halve();
        Halve::merge([x0.shuffle(i0), x1.shuffle(i1)])
    }
}

impl Shuffle for u8x16 {
    #[inline(always)]
    #[cfg(target_feature = "ssse3")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm_shuffle_epi8(self.into(), indices.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            vqtbl1q_u8(self.into(), indices.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "ssse3"))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shuffle(self, indices: Self) -> Self {
        self.swizzle_dyn(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x64(
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x64::from_array))] x: u8x64,
        #[strategy(UniformArrayStrategy::new(0u8..64u8).prop_map(u8x64::from_array))] i: u8x64,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [i0, i1] = i.halve();
        assert_eq!(x.shuffle(i).halve(), [x0.shuffle(i0), x1.shuffle(i1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x32(
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x32::from_array))] x: u8x32,
        #[strategy(UniformArrayStrategy::new(0u8..32u8).prop_map(u8x32::from_array))] i: u8x32,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [i0, i1] = i.halve();
        assert_eq!(x.shuffle(i).halve(), [x0.shuffle(i0), x1.shuffle(i1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x16(
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x16::from_array))] x: u8x16,
        #[strategy(UniformArrayStrategy::new(0u8..16u8).prop_map(u8x16::from_array))] i: u8x16,
    ) {
        assert_eq!(x.shuffle(i), x.swizzle_dyn(i));
    }
}
