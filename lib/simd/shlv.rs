use std::simd::prelude::*;

/// Trait for [`Simd<_, _>` ] types that can shift left dynamically.
pub trait Shlv {
    /// Shifts left by `shift`.
    fn shlv(self, shift: Self) -> Self;
}

impl Shlv for u16x64 {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;

            let [x0, x1] = self.halve();
            let [s0, s1] = shift.halve();
            transmute::<[u16x32; 2], Self>([x0.shlv(s0), x1.shlv(s1)])
        }
    }
}

impl Shlv for u16x32 {
    #[inline(always)]
    #[cfg(target_feature = "avx512bw")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use std::arch::x86_64::*;
            _mm512_sllv_epi16(self.into(), shift.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512bw"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;

            let [x0, x1] = self.halve();
            let [s0, s1] = shift.halve();
            transmute::<[u16x16; 2], Self>([x0.shlv(s0), x1.shlv(s1)])
        }
    }
}

impl Shlv for u16x16 {
    #[inline(always)]
    #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use std::arch::x86_64::*;
            _mm256_sllv_epi16(self.into(), shift.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512bw", target_feature = "avx512vl")))]
    #[cfg(target_feature = "avx2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use std::arch::x86_64::*;

            let x0 = _mm256_unpacklo_epi16(Simd::splat(0).into(), self.into());
            let x1 = _mm256_unpackhi_epi16(Simd::splat(0).into(), self.into());
            let s0 = _mm256_unpacklo_epi16(shift.into(), Simd::splat(0).into());
            let s1 = _mm256_unpackhi_epi16(shift.into(), Simd::splat(0).into());
            let r0 = _mm256_srli_epi32(_mm256_sllv_epi32(x0, s0), 16);
            let r1 = _mm256_srli_epi32(_mm256_sllv_epi32(x1, s1), 16);
            _mm256_packus_epi32(r0, r1).into()
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512bw", target_feature = "avx512vl")))]
    #[cfg(not(target_feature = "avx2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;
            let [x0, x1] = self.halve();
            let [s0, s1] = shift.halve();
            transmute::<[u16x8; 2], Self>([x0.shlv(s0), x1.shlv(s1)])
        }
    }
}

impl Shlv for u16x8 {
    #[inline(always)]
    #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use std::arch::x86_64::*;
            _mm_sllv_epi16(self.into(), shift.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        debug_assert!(shift.simd_lt(Simd::splat(16)).all());

        unsafe {
            use std::arch::aarch64::*;
            vshlq_u16(self.into(), vreinterpretq_s16_u16(shift.into())).into()
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512bw", target_feature = "avx512vl")))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        fallback(self, shift)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback(x: u16x8, shift: u16x8) -> u16x8 {
    debug_assert!(shift.simd_lt(Simd::splat(16)).all());

    x * u16x8::from_array([
        1u16 << shift[0],
        1u16 << shift[1],
        1u16 << shift[2],
        1u16 << shift[3],
        1u16 << shift[4],
        1u16 << shift[5],
        1u16 << shift[6],
        1u16 << shift[7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x64(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x64::from_array))]
        x: u16x64,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x64::from_array))] s: u16x64,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x32(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x32::from_array))]
        x: u16x32,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x32::from_array))] s: u16x32,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x16(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x16::from_array))]
        x: u16x16,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x16::from_array))] s: u16x16,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x8(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x8::from_array))] x: u16x8,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x8::from_array))] s: u16x8,
    ) {
        assert_eq!(x.shlv(s), fallback(x, s));
    }
}
