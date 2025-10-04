use crate::{simd::*, util::Aligned};
use bytemuck::Zeroable;
use std::mem::transmute;
use std::ops::{Mul, Shr};

/// The output layer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Output<const N: usize> {
    #[cfg_attr(test, map(|b: i8| i32::from(b)))]
    pub bias: i32,
    #[cfg_attr(test, map(|vs: [[i8; N]; 2]| Aligned(vs.map(|v| v.map(i16::from)))))]
    pub weight: Aligned<[[i16; N]; 2]>,
}

impl<const N: usize> Output<N> {
    /// Transforms the accumulator.
    #[inline(always)]
    pub fn forward(&self, us: &Aligned<[i16; N]>, them: &Aligned<[i16; N]>) -> i32 {
        const { assert!(N.is_multiple_of(128)) }

        unsafe {
            #[cfg(any(target_feature = "avx512f", target_feature = "sme"))]
            const REGISTERS: usize = 4;

            #[cfg(not(any(target_feature = "avx512f", target_feature = "sme")))]
            const REGISTERS: usize = 2;

            let mut y = [Simd::splat(0); REGISTERS];
            for (w, x) in self.weight.iter().zip([us, them]) {
                let w = transmute::<&[[i16; 128]], &[[i16x32; 4]]>(w.as_chunks_unchecked::<128>());
                let x = transmute::<&[[i16; 128]], &[[i16x32; 4]]>(x.as_chunks_unchecked::<128>());

                for (w, x) in w.iter().zip(x) {
                    let p0 = x[0].simd_clamp(Simd::splat(0), Simd::splat(255));
                    let p1 = x[1].simd_clamp(Simd::splat(0), Simd::splat(255));
                    let p2 = x[2].simd_clamp(Simd::splat(0), Simd::splat(255));
                    let p3 = x[3].simd_clamp(Simd::splat(0), Simd::splat(255));
                    y[0 % REGISTERS] = p0.mul(w[0]).mul_add_2x16(p0, y[0 % REGISTERS]);
                    y[1 % REGISTERS] = p1.mul(w[1]).mul_add_2x16(p1, y[1 % REGISTERS]);
                    y[2 % REGISTERS] = p2.mul(w[2]).mul_add_2x16(p2, y[2 % REGISTERS]);
                    y[3 % REGISTERS] = p3.mul(w[3]).mul_add_2x16(p3, y[3 % REGISTERS]);
                }
            }

            self.bias + y.iter().sum::<i32x16>().reduce_sum().shr(9)
        }
    }
}
