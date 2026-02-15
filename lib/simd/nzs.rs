use crate::simd::{Aligned, V2, W2};
use crate::util::Assume;
use std::simd::prelude::*;

/// Trait for [`Simd<u32, _>` ] types that implement `nnz`.
pub trait Nzs<const N: usize>: AsMut<[u16]> {
    /// Fills `self` with indices to non-zero_elements.
    fn nzs(&mut self, ns: &[[V2<u32>; 2]; N]) -> usize;
}

impl<const M: usize, const N: usize> Nzs<N> for Aligned<[u16; M]> {
    #[inline(always)]
    #[cfg(target_feature = "avx512vbmi2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nzs(&mut self, ns: &[[u32x16; 2]; N]) -> usize {
        const { assert!(M == N * 2 * W2) }

        use std::{arch::x86_64::*, mem::transmute};

        let mut len = 0;
        let mut base = u16x32::from_array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ]);

        for [n0, n1] in ns {
            let mask0 = n0.simd_gt(Simd::splat(0)).to_bitmask();
            let mask1 = n1.simd_gt(Simd::splat(0)).to_bitmask();
            let mask = unsafe { _mm512_kunpackw(mask1 as u32, mask0 as u32) };
            let count = mask.count_ones() as usize;

            if count > 0 {
                let indices = unsafe {
                    transmute::<__m512i, u16x32>(_mm512_maskz_compress_epi16(
                        mask,
                        transmute::<u16x32, __m512i>(base),
                    ))
                };

                let slice = self.get_mut(len..).assume();
                (slice.len() >= u16x32::LEN).assume();
                indices.copy_to_slice(slice);
                len += count;
            }

            base += u16x32::splat(32);
        }

        len
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn nzs(&mut self, ns: &[[V2<u32>; 2]; N]) -> usize {
        const { assert!(M == N * 2 * W2) }
        const NNZ_OFFSETS: Aligned<[u16x8; 256]> = {
            let mut offsets = Aligned([u16x8::splat(0); 256]);
            let table: &mut [[u16; 8]; 256] = offsets.cast_mut();

            let mut i = 0;
            while i < 256 {
                let mut j = i;
                let mut k = 0;
                while j != 0 {
                    table[i][k] = j.trailing_zeros() as u16;
                    j &= j - 1;
                    k += 1;
                }
                i += 1;
            }

            offsets
        };

        let mut len = 0;
        let mut base = u16x8::splat(0);

        for [b0, b1] in ns {
            let mut mask = b0.simd_gt(Simd::splat(0)).to_bitmask();
            mask |= b1.simd_gt(Simd::splat(0)).to_bitmask() << W2;
            for i in 0..2 * W2 / 8 {
                let idx = (mask >> (i * 8)) & 0xFF;
                let indices = base + NNZ_OFFSETS.get(idx as usize).assume();
                let slice = self.get_mut(len..).assume();
                (slice.len() >= u16x8::LEN).assume();
                indices.copy_to_slice(slice);
                len += idx.count_ones() as usize;
                base += u16x8::splat(8);
            }
        }

        len
    }
}
