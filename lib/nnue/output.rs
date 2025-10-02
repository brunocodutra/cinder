use crate::nnue::{HLQ, Layer, Layer3, Synapse};
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
        unsafe {
            let is = transmute::<&[i32; N], &[R2<i32>; N / W2]>(input)
                .map(|i| i.simd_clamp(Simd::splat(0), Simd::splat(HLQ)).cast::<u8>());

            const W: usize = usize::min(W8, N);
            const K: usize = usize::max(N / W, 1);
            let ws = transmute::<&[i8; N], &[Simd<i8, W>; K]>(&self.weight);
            let xs = transmute::<&[R2<u8>; N / W2], &[Simd<u8, W>; K]>(&is);
            let ys: [_; K] = array::from_fn(|k| ws[k].mul_4x8(xs[k]));
            self.bias + ys.iter().sum::<Simd<i32, { W / 4 }>>().reduce_sum()
        }
    }
}
