use crate::nnue::{Layer, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::{array::from_fn as each, mem::transmute};

const N: usize = Layer3::LEN;

/// The output layer.
#[derive(Debug, Zeroable)]
pub struct Output {
    pub bias: Aligned<[f32; N]>,
    pub weight: Aligned<[f32; N]>,
}

impl Synapse for Output {
    type Input<'a> = Layer3<'a>;
    type Output = V2<f32>;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(N.is_multiple_of(W2)) }

        unsafe {
            let active = transmute::<&[f32; N], &[V2<f32>; N / W2]>(input)
                .map(|i| i.simd_clamp(Simd::splat(0.), Simd::splat(1.)).powi::<2>());

            let bs = transmute::<&[f32; N], &[V2<f32>; N / W2]>(&self.bias);
            let ws = transmute::<&[f32; N], &[V2<f32>; N / W2]>(&self.weight);
            let output: [_; N / W2] = each(|i| ws[i].mul_add(active[i], bs[i]));
            output.iter().sum::<V2<f32>>()
        }
    }
}
