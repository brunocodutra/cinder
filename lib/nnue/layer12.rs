use crate::nnue::{FTQ, HLS, Layer, Layer1, Layer2, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::mem::{transmute, transmute_copy};
use std::{array::from_fn as each, ops::Mul};

const I: usize = Layer1::LEN;
const O: usize = Layer2::LEN;

const I2F: f32 = (1 << 9) as f32 / (FTQ as f32 * FTQ as f32 * HLS as f32);

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
#[derive(Debug, Zeroable)]
pub struct Layer12<S> {
    pub bypass: Aligned<[f32; O]>,
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Layer2<'a>, Output = V2<f32>>> Synapse for Layer12<S> {
    type Input<'a> = Layer1<'a>;
    type Output = i32;

    #[inline(always)]
    fn forward<'a>(&self, (us, them): Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(2 * W8)) }
        const { assert!(O.is_multiple_of(W2)) }

        unsafe {
            let us = transmute::<&[i16; I], &[[V4<i16>; I / W8]; 2]>(us);
            let them = transmute::<&[i16; I], &[[V4<i16>; I / W8]; 2]>(them);

            let active: [V8<u8>; I / W8] = each(|i| {
                let xl0 = us[0][i].simd_min(Simd::splat(FTQ));
                let xl1 = them[0][i].simd_min(Simd::splat(FTQ));
                let xh0 = us[1][i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
                let xh1 = them[1][i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
                xl0.mul_high::<9>(xh0).pack(xl1.mul_high::<9>(xh1))
            });

            let mut nzs_len = 0;
            let mut nzs = [0u16; I / 4];
            let mut base = u16x8::splat(0);

            for [b0, b1] in transmute::<&[V8<u8>; I / W8], &[[V2<u32>; 2]; I / W2 / 8]>(&active) {
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
            let mut acc = [[Simd::splat(0); K]; O / W2];
            let mut nzs = nzs[..nzs_len].iter().copied().array_chunks::<K>();
            let xs = transmute::<&[V8<u8>; I / W8], &[u8x4; I / 4]>(&active);
            let ws = transmute::<&[[i8; 4]; I * O / 4], &[[V8<i8>; O / W2]; I / 4]>(&self.weight);

            for nzs in &mut nzs {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = each(|k| {
                        let x = xs.get_unchecked(nzs[k] as usize);
                        let w = ws.get_unchecked(nzs[k] as usize)[j];
                        w.mul_add_4x8(transmute_copy::<[u8x4; W2], V8<u8>>(&[*x; W2]), a[k])
                    });
                }
            }

            for nz in nzs.into_remainder().into_iter().flatten() {
                let x = transmute_copy::<[u8x4; W2], V8<u8>>(&[*xs.get_unchecked(nz as usize); W2]);
                for (j, a) in acc.iter_mut().enumerate() {
                    a[0] = ws.get_unchecked(nz as usize)[j].mul_add_4x8(x, a[0]);
                }
            }

            let mut output = self.bias;
            let os = transmute::<&mut [f32; O], &mut [V2<f32>; O / W2]>(&mut output);

            *os = each(|j| {
                let sum = acc[j].iter().sum::<V2<i32>>();
                sum.cast::<f32>().mul_add(Simd::splat(I2F), os[j])
            });

            let bps = transmute::<&[f32; O], &[V2<f32>; O / W2]>(&self.bypass);
            let res: [_; O / W2] = each(|i| bps[i] * os[i]);
            let output = res.iter().sum::<V2<f32>>() + self.next.forward(&output);
            output.reduce_sum().mul(HLS as f32).round_ties_even() as _
        }
    }
}
