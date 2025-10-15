use crate::nnue::{HLS, Layer, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::{array::from_fn as each, mem::transmute, ops::Mul};

const I: usize = Layer3::LEN;
const O: usize = Layer3::LEN;

/// The output layer.
#[derive(Debug, Zeroable)]
pub struct Output {
    pub bias: f32,
    pub weight: Aligned<[f32; I * O]>,
}

impl Synapse for Output {
    type Input<'a> = Layer3<'a>;
    type Output = i32;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }
        const { assert!(O.is_multiple_of(W2)) }

        unsafe {
            let is = transmute::<&[f32; I], &[R2<f32>; I / W2]>(input)
                .map(|i| i.simd_clamp(Simd::splat(0.), Simd::splat(1.)));

            const K: usize = usize::min(8, I);
            let mut acc = [Simd::splat(0.); K];
            let xs = transmute::<&[R2<f32>; I / W2], &[[f32; K]; I / K]>(&is);
            let ys = transmute::<&[R2<f32>; I / W2], &[R2<f32>; O / W2]>(&is);
            let ws = transmute::<&[f32; I * O], &[[[R2<f32>; O / W2]; K]; I / K]>(&self.weight);

            for (i, xs) in xs.iter().enumerate() {
                for (j, ys) in ys.iter().enumerate() {
                    acc = each(|k| ws[i][k][j].mul_add(Simd::splat(xs[k]) * ys, acc[k]))
                }
            }

            let output = self.bias + acc.iter().sum::<R2<f32>>().reduce_sum();
            output.mul(HLS as f32) as _
        }
    }
}
