use crate::nnue::{Layer, Ln, Synapse};
use crate::simd::*;
use bytemuck::Zeroable;
use std::array;

const I: usize = Ln::LEN;

/// The output connection.
#[derive(Debug, Zeroable)]
pub struct Lno {
    pub bias: Aligned<[f32; I]>,
    pub weight: Aligned<[f32; I]>,
}

impl Synapse for Lno {
    type Input<'a> = Ln<'a>;
    type Output = V2<f32>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward(&self, input: Self::Input<'_>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }

        let xs: &[V2<f32>; I / W2] = input.as_ref();
        let ws: &[V2<f32>; I / W2] = self.weight.as_ref();
        let bs: &[V2<f32>; I / W2] = self.bias.as_ref();
        let output: [_; I / W2] = array::from_fn(|i| ws[i].mul_add(xs[i], bs[i]));
        output.iter().sum::<V2<f32>>()
    }
}
