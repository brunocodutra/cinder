use crate::nnue::{Accumulator, Feature, Layer};
use crate::util::{Aligned, Assume, Integer};
use bytemuck::Zeroable;
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;

const N: usize = Accumulator::LEN;

/// The NNUE feature transformer.
#[derive(Debug, Zeroable)]
#[debug("Transformer<{N}>")]
pub struct Transformer {
    pub bias: Aligned<[i16; N]>,
    pub weight: Aligned<[[i16; N]; Feature::LEN]>,
}

impl Transformer {
    /// Refreshes `accumulator`.
    #[inline(always)]
    pub fn refresh(&self, accumulator: &mut Aligned<[i16; N]>) {
        *accumulator = self.bias;
    }

    /// Updates `acc` by adding and removing features.
    #[inline(always)]
    pub fn accumulate_in_place(
        &self,
        acc: &mut Aligned<[i16; N]>,
        sub: [Option<Feature>; 2],
        add: [Option<Feature>; 2],
    ) {
        match (sub, add) {
            ([Some(s1), None], [None, None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i];
                }
            }

            ([None, None], [Some(a1), None]) => {
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i];
                }
            }

            ([Some(s1), Some(s2)], [None, None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i] + s2[i];
                }
            }

            ([None, None], [Some(a1), Some(a2)]) => {
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] + a2[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i];
                }
            }

            ([Some(s1), None], [Some(a1), Some(a2)]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }

    /// Updates `dst` by adding and removing features from `src`.
    #[inline(always)]
    pub fn accumulate(
        &self,
        src: &Aligned<[i16; N]>,
        dst: &mut Aligned<[i16; N]>,
        sub: [Option<Feature>; 2],
        add: [Option<Feature>; 2],
    ) {
        match (sub, add) {
            ([None, None], [None, None]) => {
                *dst = *src;
            }

            ([None, None], [Some(a1), None]) => {
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }
}
