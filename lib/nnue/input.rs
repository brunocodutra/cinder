use crate::nnue::{FTQ, HLS, L1, Layer, Ln, Synapse};
use crate::{simd::*, util::Assume};
use bytemuck::Zeroable;
use std::{array, ops::Mul};

const I: usize = L1::LEN;
const O: usize = Ln::LEN / 2;

const I2F: f32 = (1 << 9) as f32 / (FTQ as f32 * FTQ as f32 * HLS as f32);

/// The input connection.
#[derive(Debug, Zeroable)]
pub struct Input<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Ln<'a>, Output = V2<f32>>> Synapse for Input<S> {
    type Input<'a> = L1<'a>;
    type Output = f32;

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

        let mut nzs = Aligned([0u16; I / 4]);
        let nzs_len = nzs.nzs(active.cast::<[[V2<u32>; 2]; I / W2 / 8]>());

        let xs: &[u8x4; I / 4] = active.cast();
        let ws: &[[V8<i8>; O / W2]; I / 4] = self.weight.cast();
        let bs: &[V2<f32>; O / W2] = self.bias.cast();

        #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
        const K: usize = 4;

        #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
        const K: usize = 2;

        let mut acc = Aligned([[Simd::splat(0); K]; O / W2]);

        (nzs_len <= nzs.len()).assume();
        for i in (0..nzs_len - nzs_len % K).step_by(K) {
            let nz0 = *nzs.get(i).assume() as usize;
            let nz1 = *nzs.get(i + 1).assume() as usize;

            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let nz2 = *nzs.get(i + 2).assume() as usize;
            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let nz3 = *nzs.get(i + 3).assume() as usize;

            let x0 = *Aligned([*xs.get(nz0).assume(); W2]).cast();
            let x1 = *Aligned([*xs.get(nz1).assume(); W2]).cast();

            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let x2 = *Aligned([*xs.get(nz2).assume(); W2]).cast();
            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let x3 = *Aligned([*xs.get(nz3).assume(); W2]).cast();

            for (j, a) in acc.iter_mut().enumerate() {
                {
                    a[0] = ws.get(nz0).assume()[j].mul_add_4x8(x0, a[0]);
                    a[1] = ws.get(nz1).assume()[j].mul_add_4x8(x1, a[1]);
                }

                #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
                {
                    a[2] = ws.get(nz2).assume()[j].mul_add_4x8(x2, a[2]);
                    a[3] = ws.get(nz3).assume()[j].mul_add_4x8(x3, a[3]);
                }
            }
        }

        for &nz in nzs.get(nzs_len - nzs_len % K..nzs_len).assume() {
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
        result.mul(HLS as f32)
    }
}
