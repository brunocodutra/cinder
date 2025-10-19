use crate::nnue::{Layer, Layer4, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::array::from_fn as each;

const N: usize = Layer4::LEN;

/// The output layer.
#[derive(Debug, Zeroable)]
pub struct Layer4o {
    pub bias: Aligned<[f32; N]>,
    pub weight: Aligned<[f32; N]>,
}

impl Synapse for Layer4o {
    type Input<'a> = Layer4<'a>;
    type Output = V2<f32>;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(N.is_multiple_of(W2)) }

        let active = Aligned(input.cast::<[V2<f32>; N / W2]>().map(|i| {
            const ZERO: V2<f32> = Simd::splat(0.);
            const ONE: V2<f32> = Simd::splat(1.);
            i.simd_clamp(ZERO, ONE).powi::<2>()
        }));

        let bs: &[V2<f32>; N / W2] = self.bias.cast();
        let ws: &[V2<f32>; N / W2] = self.weight.cast();
        let output: [_; N / W2] = each(|i| ws[i].mul_add(active[i], bs[i]));
        output.iter().sum::<V2<f32>>()
    }
}
