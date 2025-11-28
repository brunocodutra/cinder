use crate::nnue::{FTQ, HLS, L1, Layer, Ln, Synapse, Value};
use crate::simd::*;
use crate::util::{Aligned, Assume, Float};
use bytemuck::Zeroable;
use std::{array::from_fn as each, ops::Mul};

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
    fn forward<'a>(&self, (us, them): Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(2 * W8)) }
        const { assert!(O.is_multiple_of(W2)) }

        let us: &[[V4<i16>; I / W8]; 2] = us.cast();
        let them: &[[V4<i16>; I / W8]; 2] = them.cast();

        let active: Aligned<[V8<u8>; I / W8]> = Aligned(each(|i| {
            let xl0 = us[0][i].simd_min(Simd::splat(FTQ));
            let xl1 = them[0][i].simd_min(Simd::splat(FTQ));
            let xh0 = us[1][i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
            let xh1 = them[1][i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
            xl0.mul_high::<9>(xh0).pack(xl1.mul_high::<9>(xh1))
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
                indices.copy_to_slice(nzs.get_mut(nzs_len..).assume());
                nzs_len += idx.count_ones() as usize;
                base += u16x8::splat(8);
            }
        }

        const K: usize = usize::max(8 * W2 / O, 1);
        let mut acc = Aligned([[Simd::splat(0); K]; O / W2]);
        let xs: &[u8x4; I / 4] = active.cast();
        let ws: &[[V8<i8>; O / W2]; I / 4] = self.weight.cast();
        let mut nzs = nzs[..nzs_len].iter().copied().array_chunks::<K>();

        for nzs in &mut nzs {
            for (j, a) in acc.iter_mut().enumerate() {
                *a = each(|k| {
                    let x = xs.get(nzs[k] as usize).assume();
                    let w = ws.get(nzs[k] as usize).assume()[j];
                    w.mul_add_4x8(*Aligned([*x; W2]).cast(), a[k])
                });
            }
        }

        for nz in nzs.into_remainder() {
            let x = xs.get(nz as usize).assume();
            for (j, a) in acc.iter_mut().enumerate() {
                let w = ws.get(nz as usize).assume()[j];
                a[0] = w.mul_add_4x8(*Aligned([*x; W2]).cast(), a[0]);
            }
        }

        let mut output = self.bias;
        let os: &mut [V2<f32>; O / W2] = output.cast_mut();

        *os = each(|j| {
            let sum = acc[j].iter().sum::<V2<i32>>();
            sum.cast::<f32>().mul_add(Simd::splat(I2F), os[j])
        });

        let active = Aligned([
            os.map(|i| i.simd_max(Simd::splat(0.)).powi::<2>()),
            os.map(|i| i.simd_min(Simd::splat(0.)).powi::<2>()),
        ]);

        let result = self.next.forward(active.cast()).reduce_sum();
        result.mul(HLS as f32).round_ties_even().to_int()
    }
}
