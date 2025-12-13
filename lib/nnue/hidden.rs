use crate::nnue::{Layer, Ln, Synapse};
use crate::simd::*;
use bytemuck::Zeroable;
use std::array::from_fn as each;

const I: usize = Ln::LEN;
const O: usize = Ln::LEN / 2;

/// A hidden connection.
#[derive(Debug, Zeroable)]
pub struct Hidden<S> {
    pub bias: Aligned<[f32; O]>,
    pub weight: Aligned<[[f32; 1]; I * O]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Ln<'a>>> Synapse for Hidden<S> {
    type Input<'a> = Ln<'a>;
    type Output = S::Output;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(W2)) }
        const { assert!(O.is_multiple_of(W2)) }

        const K: usize = usize::max(8 * W2 / O, 1);
        let mut acc = [[Simd::splat(0.); K]; O / W2];
        let xs: &[[f32; K]; I / K] = input.cast();
        let ws: &[[[V2<f32>; O / W2]; K]; I / K] = self.weight.cast();

        for (i, xs) in xs.iter().enumerate() {
            acc = each(|j| each(|k| ws[i][k][j].mul_add(Simd::splat(xs[k]), acc[j][k])));
        }

        let mut output = self.bias;
        let os: &mut [V2<f32>; O / W2] = output.cast_mut();
        *os = each(|j| os[j] + acc[j].iter().sum::<V2<f32>>());

        let active = Aligned([
            os.map(|i| i.simd_max(Simd::splat(0.)).powi::<2>()),
            os.map(|i| i.simd_min(Simd::splat(0.)).powi::<2>()),
        ]);

        self.next.forward(active.cast())
    }
}
