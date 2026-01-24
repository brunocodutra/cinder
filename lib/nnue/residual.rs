use crate::nnue::{Layer, Ln, Synapse};
use crate::simd::*;
use bytemuck::Zeroable;
use std::array;

const N: usize = Ln::LEN;

/// A residual connection.
#[derive(Debug, Zeroable)]
pub struct Residual<S> {
    pub weight: Aligned<[f32; N]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Ln<'a>, Output = V2<f32>>> Synapse for Residual<S> {
    type Input<'a> = Ln<'a>;
    type Output = V2<f32>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward(&self, input: Self::Input<'_>) -> Self::Output {
        const { assert!(N.is_multiple_of(W2)) }

        let xs: &[V2<f32>; N / W2] = input.cast();
        let ws: &[V2<f32>; N / W2] = self.weight.cast();
        let res: [_; N / W2] = array::from_fn(|i| ws[i] * xs[i]);
        res.iter().sum::<V2<f32>>() + self.next.forward(input)
    }
}
