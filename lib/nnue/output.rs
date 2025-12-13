use crate::nnue::{Layer, Ln, Synapse};
use crate::simd::*;
use bytemuck::Zeroable;
use std::array::from_fn as each;

const N: usize = Ln::LEN;

/// The output connection.
#[derive(Debug, Zeroable)]
pub struct Output {
    pub bias: Aligned<[f32; N]>,
    pub weight: Aligned<[f32; N]>,
}

impl Synapse for Output {
    type Input<'a> = Ln<'a>;
    type Output = V2<f32>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(N.is_multiple_of(W2)) }

        let xs: &[V2<f32>; N / W2] = input.cast();
        let ws: &[V2<f32>; N / W2] = self.weight.cast();
        let bs: &[V2<f32>; N / W2] = self.bias.cast();
        let output: [_; N / W2] = each(|i| ws[i].mul_add(xs[i], bs[i]));
        output.iter().sum::<V2<f32>>()
    }
}
