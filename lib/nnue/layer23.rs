use crate::nnue::{Layer, Layer2, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::array;
use std::mem::transmute;
use std::ops::{Add, Mul};

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
            let is = transmute::<&[f32; I], &[R2<f32>; I / W2]>(input);

            let is = [
                is.map(|i| i.mul(i).simd_clamp(Simd::splat(0.), Simd::splat(1.))),
                is.map(|i| i.simd_clamp(Simd::splat(0.), Simd::splat(1.))),
            ];

            const K: usize = usize::max(8 * W2 / O, 1);
            let mut accumulators = [[Simd::splat(0.); K]; O / W2];
            let xs = transmute::<&[[R2<f32>; I / W2]; 2], &[f32; 2 * I]>(&is);
            let ws = transmute::<&[[f32; 1]; 2 * I * O], &[R2<f32>; 2 * I * O / W2]>(&self.weight);
            for (i, xs) in xs.iter().array_chunks::<K>().enumerate() {
                let xs: [_; K] = array::from_fn(|k| Simd::splat(*xs[k]));
                for (j, acc) in accumulators.iter_mut().enumerate() {
                    *acc = array::from_fn(|k| ws[(K * i + k) * O / W2 + j].mul_add(xs[k], acc[k]));
                }
            }

            let mut output = self.bias;
            let os = transmute::<&mut [f32; O], &mut [R2<f32>; O / W2]>(&mut output);
            for (o, acc) in os.iter_mut().zip(accumulators) {
                *o = acc.iter().sum::<R2<f32>>().add(*o);
            }

            self.next.forward(&output)
        }
    }
}
