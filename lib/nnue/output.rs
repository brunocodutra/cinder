use crate::nnue::{Layer, Layer3, Synapse};
use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::array;
use std::mem::transmute;

const N: usize = Layer3::LEN;

/// The output layer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Output {
    #[cfg_attr(test, map(|b: i16| i32::from(b)))]
    pub bias: i32,
    pub weight: Aligned<[i8; N]>,
}

impl Synapse for Output {
    type Input<'a> = Layer3<'a>;
    type Output = i32;

    #[inline(always)]
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output {
        const { assert!(N.is_multiple_of(W4)) }

        unsafe {
            let is = transmute::<&[i32; N], &[R2<i32>; N / W2]>(input)
                .map(|i| i.simd_clamp(Simd::splat(0), Simd::splat(127)).cast::<i8>());

            const K: usize = usize::max(N / W4, 1);
            let ws = transmute::<&[i8; N], &[R4<i8>; K]>(&self.weight);
            let xs = transmute::<&[R2<i8>; N / W2], &[R4<i8>; K]>(&is);
            let ys: [_; K] = array::from_fn(|k| ws[k].mul_4x8(xs[k]));
            self.bias + ys.iter().sum::<R1<i32>>().reduce_sum()
        }
    }
}
