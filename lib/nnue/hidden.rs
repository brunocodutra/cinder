use crate::nnue::{Layer, Ln, Synapse};
use crate::simd::*;
use bytemuck::Zeroable;
use std::array;

const I: usize = Ln::LEN;
const O: usize = Ln::LEN / 2;

/// A hidden connection.
#[derive(Debug, Zeroable)]
pub struct Hidden<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[f32; 1]; I * O]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Ln<'a>>> Synapse for Hidden<S> {
    type Input<'a> = Ln<'a>;
    type Output = S::Output;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward(&self, input: Self::Input<'_>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }
        const { assert!(O.is_multiple_of(W2)) }

        #[cfg(target_feature = "avx512f")]
        const K: usize = 8;

        #[cfg(not(target_feature = "avx512f"))]
        #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
        const K: usize = 4;

        #[cfg(not(target_feature = "avx512f"))]
        #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
        const K: usize = 2;

        const { assert!(I.is_multiple_of(K)) }
        let xs: &[[f32; K]; I / K] = input.cast();
        let ws: &[[[V2<f32>; O / W2]; K]; I / K] = self.weight.cast();
        let bs: &[V2<f32>; O / W2] = self.bias.cast();

        let mut acc = [[Simd::splat(0.); K]; O / W2];

        for (i, xs) in xs.iter().enumerate() {
            let x0 = Simd::splat(xs[0]);
            let x1 = Simd::splat(xs[1]);

            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let x2 = Simd::splat(xs[2]);
            #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
            let x3 = Simd::splat(xs[3]);

            #[cfg(target_feature = "avx512f")]
            let x4 = Simd::splat(xs[4]);
            #[cfg(target_feature = "avx512f")]
            let x5 = Simd::splat(xs[5]);
            #[cfg(target_feature = "avx512f")]
            let x6 = Simd::splat(xs[6]);
            #[cfg(target_feature = "avx512f")]
            let x7 = Simd::splat(xs[7]);

            for (j, a) in acc.iter_mut().enumerate() {
                {
                    a[0] = ws[i][0][j].mul_add(x0, a[0]);
                    a[1] = ws[i][1][j].mul_add(x1, a[1]);
                }

                #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
                {
                    a[2] = ws[i][2][j].mul_add(x2, a[2]);
                    a[3] = ws[i][3][j].mul_add(x3, a[3]);
                }

                #[cfg(target_feature = "avx512f")]
                {
                    a[4] = ws[i][4][j].mul_add(x4, a[4]);
                    a[5] = ws[i][5][j].mul_add(x5, a[5]);
                    a[6] = ws[i][6][j].mul_add(x6, a[6]);
                    a[7] = ws[i][7][j].mul_add(x7, a[7]);
                }
            }
        }

        let output: Aligned<[V2<f32>; O / W2]> =
            Aligned(array::from_fn(|j| bs[j] + acc[j].iter().sum::<V2<f32>>()));

        let active = Aligned([
            output.map(|i| i.simd_max(Simd::splat(0.)).powi::<2>()),
            output.map(|i| i.simd_min(Simd::splat(0.)).powi::<2>()),
        ]);

        self.next.forward(active.cast())
    }
}
