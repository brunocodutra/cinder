use crate::nnue::{HLS, Layer, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::{array, mem::transmute, ops::Mul};

const N: usize = Layer3::LEN;

/// The output layer.
#[derive(Debug, Zeroable)]
pub struct Output {
    pub bias: f32,
    pub weight: Aligned<[f32; N]>,
}

impl Synapse for Output {
    type Input<'a> = Layer3<'a>;
    type Output = i32;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(N.is_multiple_of(W2)) }

        unsafe {
            let xs = transmute::<&[f32; N], &[R2<f32>; N / W2]>(input).map(|i| {
                let clipped = i.simd_clamp(Simd::splat(0.), Simd::splat(1.));
                clipped * clipped
            });

            let ws = transmute::<&[f32; N], &[R2<f32>; N / W2]>(&self.weight);
            let accumulators: [_; N / W2] = array::from_fn(|i| ws[i] * xs[i]);
            let output = self.bias + accumulators.iter().sum::<R2<f32>>().reduce_sum();
            output.mul(HLS as f32) as _
        }
    }
}
