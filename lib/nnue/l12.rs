use crate::nnue::{FTQ, I2F, L1, L2, Layer, Synapse};
use crate::simd::*;
use crate::util::{Assume, Num, StaticSeq};
use bytemuck::{Zeroable, zeroed};
use std::array;

#[cfg(feature = "permuter")]
use crate::nnue::sparsity::ActivationHistory;

const I: usize = L1::LEN;
const O: usize = L2::LEN;

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn nzs(ns: &[V2<u32>; I / W2 / 4]) -> StaticSeq<u8, { I / 4 }> {
    #[cfg(target_feature = "avx512vbmi2")]
    type Iota = u8x64;

    #[cfg(not(target_feature = "avx512vbmi2"))]
    type Iota = u8x8;

    const K: usize = Iota::LEN / W2;
    const { assert!(K * W2 <= u64::BITS as usize) }

    let mut nzs = StaticSeq::new();
    let mut iota = Iota::from_array(array::from_fn(Num::cast));
    let (chunks, _) = ns.as_chunks::<K>();

    for chunk in chunks {
        let mut mask = 0u64;
        for (i, n) in chunk.iter().enumerate() {
            mask |= n.simd_gt(zeroed()).to_bitmask() << (i * W2);
        }

        nzs.extend_from_simd_masked(iota, mask.cast());
        iota += Iota::splat(Iota::LEN as u8);
    }

    nzs
}

/// The input connection.
#[derive(Debug, Zeroable)]
pub struct L12<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = L2<'a>, Output = f32>> Synapse for L12<S> {
    type Input<'a> = L1<'a>;
    type Output = f32;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward(&self, (us, them): Self::Input<'_>) -> Self::Output {
        const { assert!(I.is_multiple_of(2 * W8)) }
        const { assert!(O.is_multiple_of(W2)) }

        let us: &[[V4<i16>; I / W8]; 2] = us.cast_ref();
        let them: &[[V4<i16>; I / W8]; 2] = them.cast_ref();

        let active: Aligned<[[V8<u8>; 2]; I / W8 / 2]> = Aligned(array::from_fn(|i| {
            let xl00 = us[0][2 * i].simd_min(Simd::splat(FTQ));
            let xl01 = us[0][2 * i + 1].simd_min(Simd::splat(FTQ));
            let xh00 = us[1][2 * i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
            let xh01 = us[1][2 * i + 1].simd_clamp(Simd::splat(0), Simd::splat(FTQ));

            let x00 = xl00.mul_high::<7>(xh00);
            let x01 = xl01.mul_high::<7>(xh01);

            let xl10 = them[0][2 * i].simd_min(Simd::splat(FTQ));
            let xl11 = them[0][2 * i + 1].simd_min(Simd::splat(FTQ));
            let xh10 = them[1][2 * i].simd_clamp(Simd::splat(0), Simd::splat(FTQ));
            let xh11 = them[1][2 * i + 1].simd_clamp(Simd::splat(0), Simd::splat(FTQ));

            let x10 = xl10.mul_high::<7>(xh10);
            let x11 = xl11.mul_high::<7>(xh11);

            [Pack::pack(x00, x10), Pack::pack(x01, x11)]
        }));

        #[cfg(feature = "permuter")]
        ActivationHistory::record(active.cast_ref());

        let xs: &[u8x4; I / 4] = active.cast_ref();
        let ws: &[[V8<i8>; O / W2]; I / 4] = self.weight.cast_ref();
        let bs: &[V2<f32>; O / W2] = self.bias.cast_ref();

        const K: usize = unroll(O / W2);
        let mut acc = Aligned([[Simd::splat(0); K]; O / W2]);
        let nzs = nzs(active.cast_ref::<[V2<u32>; I / W2 / 4]>());

        #[allow(clippy::modulo_one)]
        for i in (0..nzs.len() - nzs.len() % K).step_by(K) {
            let idx: [usize; K] = array::from_fn(|k| *nzs.get(i + k).assume() as usize);
            let x: [V8<u8>; K] = idx.map(|nz| Aligned([*xs.get(nz).assume(); W2]).cast());
            let w: [[V8<i8>; O / W2]; K] = idx.map(|nz| *ws.get(nz).assume());

            for (j, a) in acc.iter_mut().enumerate() {
                *a = array::from_fn(|k| w[k][j].mul_add_4x8(x[k], a[k]));
            }
        }

        #[allow(clippy::modulo_one)]
        for i in nzs.len() - nzs.len() % K..nzs.len() {
            let nz = *nzs.get(i).assume() as usize;
            let x = Aligned([*xs.get(nz).assume(); W2]);
            let w = ws.get(nz).assume();
            for (j, a) in acc.iter_mut().enumerate() {
                a[0] = w[j].mul_add_4x8(x.cast(), a[0]);
            }
        }

        let output: Aligned<[V2<f32>; O / W2]> = Aligned(array::from_fn(|j| {
            let sum = acc[j].iter().sum::<V2<i32>>();
            let out = sum.cast::<f32>().mul_add(Simd::splat(I2F), bs[j]);
            out * out.abs()
        }));

        self.next.forward(output.cast_ref())
    }
}
