use crate::nnue::{FTQ, HLS, L1, Layer, Ln, Synapse, Value};
use crate::simd::*;
use crate::util::{Assume, Float};
use bytemuck::Zeroable;
use std::{array, ops::Mul};

const I: usize = L1::LEN;
const O: usize = Ln::LEN / 2;

const I2F: f32 = (1 << 9) as f32 / (FTQ as f32 * FTQ as f32 * HLS as f32);

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

/// The input connection.
#[derive(Debug, Zeroable)]
pub struct Input<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Ln<'a>, Output = V2<f32>>> Synapse for Input<S> {
    type Input<'a> = L1<'a>;
    type Output = Value;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward(&self, (us, them): Self::Input<'_>) -> Self::Output {
        const { assert!(I.is_multiple_of(2 * W8)) }
        const { assert!(O.is_multiple_of(W2)) }

        let us: &[[V4<i16>; I / W8]; 2] = us.cast();
        let them: &[[V4<i16>; I / W8]; 2] = them.cast();

        let active: Aligned<[[V8<u8>; 2]; I / W8 / 2]> = Aligned(array::from_fn(|i| {
            let xl00 = us[0][2 * i].simd_min(Simd::splat(FTQ));
            let xl01 = us[0][2 * i + 1].simd_min(Simd::splat(FTQ));
            let xh00 = us[1][2 * i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
            let xh01 = us[1][2 * i + 1].simd_clamp(Simd::splat(0), Simd::splat(FTQ));

            let x00 = xl00.mul_high::<9>(xh00);
            let x01 = xl01.mul_high::<9>(xh01);

            let xl10 = them[0][2 * i].simd_min(Simd::splat(FTQ));
            let xl11 = them[0][2 * i + 1].simd_min(Simd::splat(FTQ));
            let xh10 = them[1][2 * i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
            let xh11 = them[1][2 * i + 1].simd_clamp(Simd::splat(0), Simd::splat(FTQ));

            let x10 = xl10.mul_high::<9>(xh10);
            let x11 = xl11.mul_high::<9>(xh11);

            [Pack::pack(x00, x10), Pack::pack(x01, x11)]
        }));

        let mut nzs_len = 0;
        let mut nzs = Aligned([0u16; I / 4]);
        let mut base = u16x8::splat(0);

        for [b0, b1] in active.cast::<[[V2<u32>; 2]; I / W2 / 8]>() {
            let mut mask = b0.simd_gt(Simd::splat(0)).to_bitmask();
            mask |= b1.simd_gt(Simd::splat(0)).to_bitmask() << W2;
            for i in 0..2 * W2 / 8 {
                let idx = (mask >> (i * 8)) & 0xFF;
                let indices = base + NNZ_OFFSETS.get(idx as usize).assume();
                let slice = nzs.get_mut(nzs_len..).assume();
                (slice.len() >= u16x8::LEN).assume();
                indices.copy_to_slice(slice);
                nzs_len += idx.count_ones() as usize;
                base += u16x8::splat(8);
            }
        }

        #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
        const K: usize = 4;

        #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
        const K: usize = 2;

        let xs: &[u8x4; I / 4] = active.cast();
        let ws: &[[V8<i8>; O / W2]; I / 4] = self.weight.cast();
        let bs: &[V2<f32>; O / W2] = self.bias.cast();

        let mut acc = Aligned([[Simd::splat(0); K]; O / W2]);
        let mut nzs = nzs[..nzs_len].iter().copied().array_chunks::<K>();

        for nzs in &mut nzs {
            let x0 = *Aligned([*xs.get(nzs[0] as usize).assume(); W2]).cast();
            let x1 = *Aligned([*xs.get(nzs[1] as usize).assume(); W2]).cast();

            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let x2 = *Aligned([*xs.get(nzs[2] as usize).assume(); W2]).cast();
            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let x3 = *Aligned([*xs.get(nzs[3] as usize).assume(); W2]).cast();

            for (j, a) in acc.iter_mut().enumerate() {
                {
                    a[0] = ws.get(nzs[0] as usize).assume()[j].mul_add_4x8(x0, a[0]);
                    a[1] = ws.get(nzs[1] as usize).assume()[j].mul_add_4x8(x1, a[1]);
                }

                #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
                {
                    a[2] = ws.get(nzs[2] as usize).assume()[j].mul_add_4x8(x2, a[2]);
                    a[3] = ws.get(nzs[3] as usize).assume()[j].mul_add_4x8(x3, a[3]);
                }
            }
        }

        for nz in nzs.into_remainder() {
            let x = *Aligned([*xs.get(nz as usize).assume(); W2]).cast();
            for (j, a) in acc.iter_mut().enumerate() {
                a[0] = ws.get(nz as usize).assume()[j].mul_add_4x8(x, a[0]);
            }
        }

        let output: Aligned<[V2<f32>; O / W2]> = Aligned(array::from_fn(|j| {
            let sum = acc[j].iter().sum::<V2<i32>>();
            sum.cast::<f32>().mul_add(Simd::splat(I2F), bs[j])
        }));

        let active = Aligned([
            output.map(|i| i.simd_max(Simd::splat(0.)).powi::<2>()),
            output.map(|i| i.simd_min(Simd::splat(0.)).powi::<2>()),
        ]);

        let result = self.next.forward(active.cast()).reduce_sum();
        result.mul(HLS as f32).round_ties_even().to_int()
    }
}
