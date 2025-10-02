use crate::nnue::{Layer, Layer1, Layer2, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::ops::{Add, Shr};
use std::{array, mem::transmute};

const I: usize = Layer1::LEN;
const O: usize = Layer2::LEN;

/// The first hidden transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Hidden1<S> {
    #[cfg_attr(test, map(|vs: [i16; O]| Aligned(vs.map(i32::from))))]
    pub bias: Aligned<[i32; O]>,
    pub weight: Aligned<[[i8; 4]; I * O / 4]>,
    pub next: S,
}

impl<S: for<'a> Synapse<Input<'a> = Layer2<'a>>> Synapse for Hidden1<S> {
    type Input<'a> = Layer1<'a>;
    type Output = S::Output;

    #[inline(always)]
    fn forward<'a>(&self, (us, them): Self::Input<'a>) -> Self::Output {
        const { assert!(I.is_multiple_of(2 * W8)) }
        const { assert!(O.is_multiple_of(W2)) }

        unsafe {
            let us = transmute::<&[i16; I], &[[R4<i16>; I / W8]; 2]>(us);
            let them = transmute::<&[i16; I], &[[R4<i16>; I / W8]; 2]>(them);

            let is: [R8<i8>; I / W8] = array::from_fn(|i| {
                let xl0 = us[0][i].simd_min(Simd::splat(255));
                let xl1 = them[0][i].simd_min(Simd::splat(255));
                let xh0 = us[1][i].simd_clamp(Simd::splat(0), Simd::splat(255));
                let xh1 = them[1][i].simd_clamp(Simd::splat(0), Simd::splat(255));
                xl0.mul_high::<9>(xh0).pack(xl1.mul_high::<9>(xh1))
            });

            const K: usize = usize::max(4 * W2 / O, 1);
            let mut accumulators = [[Simd::splat(0); K]; O / W2];
            let xs = transmute::<&[R8<i8>; I / W2 / 4], &[i8x4; I / 4]>(&is);
            let ws = transmute::<&[[i8; 4]; I * O / 4], &[R8<i8>; I * O / W2 / 4]>(&self.weight);
            for (i, xs) in xs.iter().array_chunks::<K>().enumerate() {
                for (j, acc) in accumulators.iter_mut().enumerate() {
                    *acc = array::from_fn(|k| {
                        let ws = &ws[(K * i + k) * O / W2 + j];
                        ws.mul_add_4x8(transmute::<[i8x4; W2], R8<i8>>([*xs[k]; W2]), acc[k])
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
