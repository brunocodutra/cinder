use crate::nnue::{FTQ, Layer, Layer1, Layer2, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::array;
use std::mem::{transmute, transmute_copy};
use std::ops::{Add, Shr};

const I: usize = Layer1::LEN;
const O: usize = Layer2::LEN;

const NNZ_OFFSETS: [u16x8; 256] = {
    let mut offsets = [u16x8::splat(0); 256];
    let table = unsafe { transmute::<&mut [u16x8; 256], &mut [[u16; 8]; 256]>(&mut offsets) };

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

/// The first hidden transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Layer12<S> {
    #[cfg_attr(test, map(|vs: [i16; O]| Aligned(vs.map(i32::from))))]
    pub bias: Aligned<[i32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Layer2<'a>>> Synapse for Layer12<S> {
    type Input<'a> = Layer1<'a>;
    type Output = S::Output;

    #[inline(always)]
    fn forward<'a>(&self, (us, them): Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(2 * W8)) }
        const { assert!(O.is_multiple_of(W2)) }

        unsafe {
            let us = transmute::<&[i16; I], &[[R4<i16>; I / W8]; 2]>(us);
            let them = transmute::<&[i16; I], &[[R4<i16>; I / W8]; 2]>(them);

            let is: [R8<u8>; I / W8] = array::from_fn(|i| {
                let xl0 = us[0][i].simd_min(Simd::splat(FTQ));
                let xl1 = them[0][i].simd_min(Simd::splat(FTQ));
                let xh0 = us[1][i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
                let xh1 = them[1][i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
                xl0.mul_high::<9>(xh0).pack(xl1.mul_high::<9>(xh1))
            });

            let mut nzs_len = 0;
            let mut nzs = [0u16; I / 4];
            let mut base = u16x8::splat(0);

            for [b0, b1] in transmute::<&[R8<u8>; I / W8], &[[R2<u32>; 2]; I / W2 / 8]>(&is) {
                let mut mask = b0.simd_gt(Simd::splat(0)).to_bitmask();
                mask |= b1.simd_gt(Simd::splat(0)).to_bitmask() << W2;
                for i in 0..2 * W2 / 8 {
                    let idx = (mask >> (i * 8)) & 0xFF;
                    let indices = base + NNZ_OFFSETS[idx as usize];
                    indices.copy_to_slice(&mut nzs[nzs_len..]);
                    nzs_len += idx.count_ones() as usize;
                    base += u16x8::splat(8);
                }
            }

            const K: usize = usize::max(8 * W2 / O, 1);
            let mut accumulators = [[Simd::splat(0); K]; O / W2];
            let xs = transmute::<&[R8<u8>; I / W8], &[u8x4; I / 4]>(&is);
            let ws = transmute::<&[[i8; 4]; I * O / 4], &[R8<i8>; I * O / W8]>(&self.weight);
            let mut nzs = nzs[..nzs_len].iter().copied().array_chunks::<K>();
            for nzs in &mut nzs {
                let xs = nzs.map(|nz| transmute_copy::<[u8x4; W2], R8<u8>>(&[xs[nz as usize]; W2]));
                for (j, acc) in accumulators.iter_mut().enumerate() {
                    let ws = nzs.map(|nz| ws[nz as usize * O / W2 + j]);
                    *acc = array::from_fn(|k| ws[k].mul_add_4x8(xs[k], acc[k]));
                }
            }

            let mut accumulators = accumulators.map(|acc| acc.iter().sum::<R2<i32>>());
            for nz in nzs.into_remainder().into_iter().flatten() {
                let x = transmute_copy::<[u8x4; W2], R8<u8>>(&[xs[nz as usize]; W2]);
                for (j, acc) in accumulators.iter_mut().enumerate() {
                    *acc = ws[nz as usize * O / W2 + j].mul_add_4x8(x, *acc);
                }
            }

            let mut output = self.bias;
            let os = transmute::<&mut [i32; O], &mut [R2<i32>; O / W2]>(&mut output);
            for (o, acc) in os.iter_mut().zip(accumulators) {
                *o = acc.add(*o).shr(6);
            }

            self.next.forward(&output)
        }
    }
}
