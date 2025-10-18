use crate::nnue::{Layer, Layer2, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::{array::from_fn as each, mem::transmute};

const I: usize = Layer2::LEN;
const O: usize = Layer3::LEN;

/// The second hidden transformer.
#[derive(Debug, Zeroable)]
pub struct Layer23<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[f32; 1]; 2 * I * O]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Layer3<'a>>> Synapse for Layer23<S> {
    type Input<'a> = Layer2<'a>;
    type Output = S::Output;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }
        const { assert!(O.is_multiple_of(W2)) }

        unsafe {
            let is = transmute::<&[f32; I], &[V2<f32>; I / W2]>(input);

            let active = [
                is.map(|i| i.simd_max(Simd::splat(0.)).powi::<2>()),
                is.map(|i| i.simd_min(Simd::splat(0.)).powi::<2>()),
            ];

            const K: usize = usize::max(8 * W2 / O, 1);
            let mut acc = [[Simd::splat(0.); K]; O / W2];
            let xs = transmute::<&[[V2<f32>; I / W2]; 2], &[[f32; K]; 2 * I / K]>(&active);
            let ws = transmute::<&[[f32; 1]; 2 * I * O], &[[[V2<f32>; O / W2]; K]; 2 * I / K]>(
                &self.weight,
            );

            for (i, xs) in xs.iter().enumerate() {
                acc = each(|j| each(|k| ws[i][k][j].mul_add(Simd::splat(xs[k]), acc[j][k])));
            }

            let mut output = self.bias;
            let os = transmute::<&mut [f32; O], &mut [V2<f32>; O / W2]>(&mut output);
            *os = each(|j| acc[j].iter().sum::<V2<f32>>() + os[j]);

            self.next.forward(&output)
        }
    }
}
