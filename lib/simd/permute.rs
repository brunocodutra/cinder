use crate::simd::Shuffle;
use std::simd::prelude::*;

/// Trait for [`Simd<_, _>` ] types that can permute across lanes.
pub trait Permute {
    /// Permutes across lanes given `indices`.
    fn permute(self, indices: Self) -> Self;
}

impl Permute for u8x64 {
    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm512_permutexvar_epi8(indices.into(), self.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi"))]
    #[cfg(target_feature = "avx2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use crate::simd::Halve;
            use std::arch::x86_64::*;

            let [x0, x1] = self.halve();
            let [i0, i1] = indices.halve();

            let x0l: u8x32 = _mm256_permute2x128_si256::<0x00>(x0.into(), x0.into()).into();
            let x0h: u8x32 = _mm256_permute2x128_si256::<0x11>(x0.into(), x0.into()).into();
            let x1l: u8x32 = _mm256_permute2x128_si256::<0x00>(x1.into(), x1.into()).into();
            let x1h: u8x32 = _mm256_permute2x128_si256::<0x11>(x1.into(), x1.into()).into();

            let (m0, m1) = (i0 << 3, i1 << 3);
            let (j0, j1) = (!u8x32::splat(32) & i0, !u8x32::splat(32) & i1);
            let x00 = _mm256_blendv_epi8(x0l.shuffle(j0).into(), x0h.shuffle(j0).into(), m0.into());
            let x01 = _mm256_blendv_epi8(x0l.shuffle(j1).into(), x0h.shuffle(j1).into(), m1.into());
            let x10 = _mm256_blendv_epi8(x1l.shuffle(j0).into(), x1h.shuffle(j0).into(), m0.into());
            let x11 = _mm256_blendv_epi8(x1l.shuffle(j1).into(), x1h.shuffle(j1).into(), m1.into());

            let (m0, m1) = (i0 << 2, i1 << 2);
            let y0 = _mm256_blendv_epi8(x00.into(), x10.into(), m0.into());
            let y1 = _mm256_blendv_epi8(x01.into(), x11.into(), m1.into());

            Halve::merge([y0.into(), y1.into()])
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            use std::mem::{transmute, transmute_copy};

            let x = transmute::<Self, uint8x16x4_t>(self);
            let i = transmute::<Self, uint8x16x4_t>(indices);

            let y0 = vqtbl4q_u8(x, i.0);
            let y1 = vqtbl4q_u8(x, i.1);
            let y2 = vqtbl4q_u8(x, i.2);
            let y3 = vqtbl4q_u8(x, i.3);

            transmute_copy::<[uint8x16_t; 4], Self>(&[y0, y1, y2, y3])
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi"))]
    #[cfg(not(target_feature = "avx2"))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;

            let [x0, x1] = self.halve();
            let [i0, i1] = indices.halve();

            let i0w = transmute::<u8x32, u16x16>(i0) << 2;
            let i1w = transmute::<u8x32, u16x16>(i1) << 2;

            let m0: mask8x32 = transmute::<u16x16, i8x32>(i0w).simd_lt(Simd::splat(0));
            let m1: mask8x32 = transmute::<u16x16, i8x32>(i1w).simd_lt(Simd::splat(0));

            let o0 = m0.select(x1.permute(i0 & Simd::splat(31)), x0.permute(i0));
            let o1 = m1.select(x1.permute(i1 & Simd::splat(31)), x0.permute(i1));

            Halve::merge([o0, o1])
        }
    }
}

impl Permute for u8x32 {
    #[inline(always)]
    #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use std::arch::x86_64::*;
            _mm256_permutexvar_epi8(indices.into(), self.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "avx512vl")))]
    #[cfg(target_feature = "avx2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use std::arch::x86_64::*;

            let x0: Self = _mm256_permute2x128_si256::<0x00>(self.into(), self.into()).into();
            let x1: Self = _mm256_permute2x128_si256::<0x11>(self.into(), self.into()).into();

            let y0 = x0.shuffle(indices).into();
            let y1 = x1.shuffle(indices).into();

            let mask = indices << 3;
            _mm256_blendv_epi8(y0, y1, mask.into()).into()
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi"))]
    #[cfg(not(target_feature = "avx2"))]
    #[cfg(target_feature = "sse4.1")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use crate::simd::Halve;
            use std::{arch::x86_64::*, mem::transmute_copy};

            let [x0, x1] = self.halve();
            let [i0, i1] = indices.halve();

            let (m0, m1) = (i0 << 3, i1 << 3);
            let (j0, j1) = (!u8x16::splat(16) & i0, !u8x16::splat(16) & i1);

            let y0 = _mm_blendv_epi8(x0.permute(j0).into(), x1.permute(j0).into(), m0.into());
            let y1 = _mm_blendv_epi8(x0.permute(j1).into(), x1.permute(j1).into(), m1.into());

            Halve::merge([y0.into(), y1.into()])
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "neon")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            use std::mem::{transmute, transmute_copy};

            let x = transmute::<Self, uint8x16x2_t>(self);
            let i = transmute::<Self, uint8x16x2_t>(indices);

            let y0 = vqtbl2q_u8(x, i.0);
            let y1 = vqtbl2q_u8(x, i.1);

            transmute_copy::<[uint8x16_t; 2], Self>(&[y0, y1])
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi"))]
    #[cfg(not(target_feature = "avx2"))]
    #[cfg(not(target_feature = "sse4.1"))]
    #[cfg(not(target_feature = "neon"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;

            let [x0, x1] = self.halve();
            let [i0, i1] = indices.halve();

            let i0w = transmute::<u8x16, u16x8>(i0) << 3;
            let i1w = transmute::<u8x16, u16x8>(i1) << 3;

            let m0: mask8x16 = transmute::<u16x8, i8x16>(i0w).simd_lt(Simd::splat(0));
            let m1: mask8x16 = transmute::<u16x8, i8x16>(i1w).simd_lt(Simd::splat(0));

            let o0 = m0.select(x1.permute(i0 & Simd::splat(15)), x0.permute(i0));
            let o1 = m1.select(x1.permute(i1 & Simd::splat(15)), x0.permute(i1));

            Halve::merge([o0, o1])
        }
    }
}

impl Permute for u8x16 {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn permute(self, indices: Self) -> Self {
        self.shuffle(indices)
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
        assert_eq!(x.permute(i), x.swizzle_dyn(i));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x32(
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x32::from_array))] x: u8x32,
        #[strategy(UniformArrayStrategy::new(0u8..32u8).prop_map(u8x32::from_array))] i: u8x32,
    ) {
        assert_eq!(x.permute(i), x.swizzle_dyn(i));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x16(
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x16::from_array))] x: u8x16,
        #[strategy(UniformArrayStrategy::new(0u8..16u8).prop_map(u8x16::from_array))] i: u8x16,
    ) {
        assert_eq!(x.permute(i), x.swizzle_dyn(i));
    }
}
