use crate::nnue::{Layer, Layer3, Layer4, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::array::from_fn as each;

const I: usize = Layer3::LEN;
const O: usize = Layer4::LEN;

/// The second hidden transformer.
#[derive(Debug, Zeroable)]
pub struct Layer34<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[f32; 1]; 2 * I * O]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Layer4<'a>>> Synapse for Layer34<S> {
    type Input<'a> = Layer3<'a>;
    type Output = S::Output;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }
        const { assert!(O.is_multiple_of(W2)) }

        let is: &[V2<f32>; I / W2] = input.cast();

        let active = Aligned([
            is.map(|i| i.simd_max(Simd::splat(0.)).powi::<2>()),
            is.map(|i| i.simd_min(Simd::splat(0.)).powi::<2>()),
        ]);

        const K: usize = usize::max(8 * W2 / O, 1);
        let mut acc = [[Simd::splat(0.); K]; O / W2];
        let xs: &[[f32; K]; 2 * I / K] = active.cast();
        let ws: &[[[V2<f32>; O / W2]; K]; 2 * I / K] = self.weight.cast();

        for (i, xs) in xs.iter().enumerate() {
            acc = each(|j| each(|k| ws[i][k][j].mul_add(Simd::splat(xs[k]), acc[j][k])));
        }

        let mut output = self.bias;
        let os: &mut [V2<f32>; O / W2] = output.cast_mut();
        *os = each(|j| acc[j].iter().sum::<V2<f32>>() + os[j]);

        self.next.forward(&output)
    }
}
