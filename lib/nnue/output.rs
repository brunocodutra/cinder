use crate::nnue::{Layer, Layer2, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::{array::from_fn as each, mem::transmute};

const I: usize = Layer3::LEN;
const O: usize = Layer2::LEN;

/// The output layer.
#[derive(Debug, Zeroable)]
pub struct Output {
    pub bias: Aligned<[f32; I]>,
    pub weight: Aligned<[f32; I]>,
}

impl Synapse for Output {
    type Input<'a> = Layer3<'a>;
    type Output = [V2<f32>; O / W2];

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(O)) }
        const { assert!(I.is_multiple_of(W2)) }
        const { assert!(O.is_multiple_of(W2)) }

        unsafe {
            let active = transmute::<&[f32; I], &[V2<f32>; I / W2]>(input)
                .map(|i| i.simd_clamp(Simd::splat(0.), Simd::splat(1.)).powi::<2>());

            let bs = transmute::<&[f32; I], &[[V2<f32>; I / O]; O / W2]>(&self.bias);
            let ws = transmute::<&[f32; I], &[[V2<f32>; I / O]; O / W2]>(&self.weight);
            let xs = transmute::<&[V2<f32>; I / W2], &[[V2<f32>; I / O]; O / W2]>(&active);
            let output: [_; O / W2] = each(|i| each(|j| ws[i][j].mul_add(xs[i][j], bs[i][j])));
            output.map(|a: [_; I / O]| a.iter().sum::<V2<f32>>())
        }
    }
}
