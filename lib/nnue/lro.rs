use crate::nnue::{Layer, Ln, Synapse};
use crate::simd::*;
use bytemuck::Zeroable;
use std::array;

const I: usize = Ln::LEN;

/// A residual connection.
#[derive(Debug, Zeroable)]
pub struct Lro<S> {
    pub weight: Aligned<[f32; I]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Ln<'a>, Output = V2<f32>>> Synapse for Lro<S> {
    type Input<'a> = Ln<'a>;
    type Output = V2<f32>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward(&self, input: Self::Input<'_>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }

        let xs: &[V2<f32>; I / W2] = input.as_ref();
        let ws: &[V2<f32>; I / W2] = self.weight.as_ref();
        let res: [_; I / W2] = array::from_fn(|i| ws[i] * xs[i]);
        res.iter().sum::<V2<f32>>() + self.next.forward(input)
    }
}
