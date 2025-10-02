use crate::nnue::{HLQ, Layer, Layer2, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::array;
use std::mem::{transmute, transmute_copy};
use std::ops::{Add, Shr};

const I: usize = Layer2::LEN;
const O: usize = Layer3::LEN;

/// The second hidden transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Layer23<S> {
    #[cfg_attr(test, map(|vs: [i16; O]| Aligned(vs.map(i32::from))))]
    pub bias: Aligned<[i32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
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
            let is = transmute::<&[i32; I], &[R2<i32>; I / W2]>(input)
                .map(|i| i.simd_clamp(Simd::splat(0), Simd::splat(HLQ)).cast::<u8>());

            const K: usize = usize::max(8 * W2 / O, 1);
            let mut accumulators = [[Simd::splat(0); K]; O / W2];
            let xs = transmute::<&[R2<u8>; I / W2], &[u8x4; I / 4]>(&is);
            let ws = transmute::<&[[i8; 4]; I * O / 4], &[R8<i8>; I * O / W8]>(&self.weight);
            for (i, xs) in xs.iter().array_chunks::<K>().enumerate() {
                for (j, acc) in accumulators.iter_mut().enumerate() {
                    *acc = array::from_fn(|k| {
                        let x = transmute_copy::<[u8x4; W2], R8<u8>>(&[*xs[k]; W2]);
                        ws[(K * i + k) * O / W2 + j].mul_add_4x8(x, acc[k])
                    });
                }
            }

            let mut output = self.bias;
            let os = transmute::<&mut [i32; O], &mut [R2<i32>; O / W2]>(&mut output);
            for (o, acc) in os.iter_mut().zip(accumulators) {
                *o = acc.iter().sum::<R2<i32>>().add(*o).shr(6);
            }

            self.next.forward(&output)
        }
    }
}
