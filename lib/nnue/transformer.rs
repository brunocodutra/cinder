use crate::nnue::{Accumulator, KAFeature, Layer, TIFeature};
use crate::simd::*;
use crate::util::{Assume, Num};
use bytemuck::Zeroable;
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;

const N: usize = Accumulator::LEN;

/// The NNUE feature transformer.
#[derive(Debug, Zeroable)]
#[debug("Transformer<{N}>")]
pub struct Transformer {
    pub bias: Aligned<[i16; N]>,
    pub ka: Aligned<[[i16; N]; KAFeature::LEN]>,
    pub ti: Aligned<[[i16; N]; TIFeature::LEN]>,
}

impl Transformer {
    /// Refreshes `accumulator`.
    #[inline(always)]
    pub fn refresh(&self, accumulator: &mut Aligned<[i16; N]>) {
        *accumulator = self.bias;
    }

    /// Updates `dst` by adding and removing [`KAFeature`]s from `src`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn accumulate_ka(
        &self,
        src: &Aligned<[i16; N]>,
        dst: &mut Aligned<[i16; N]>,
        sub: [Option<KAFeature>; 2],
        add: [Option<KAFeature>; 2],
    ) {
        match (sub, add) {
            ([None, None], [Some(a1), None]) => {
                let a1 = self.ka.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let s2 = self.ka.get(s2.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let s2 = self.ka.get(s2.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();
                let a2 = self.ka.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }

    /// Updates `acc` by adding and removing [`KAFeature`]s.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn accumulate_ka_in_place(
        &self,
        acc: &mut Aligned<[i16; N]>,
        sub: [Option<KAFeature>; 2],
        add: [Option<KAFeature>; 2],
    ) {
        match (sub, add) {
            ([Some(s1), None], [None, None]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i];
                }
            }

            ([None, None], [Some(a1), None]) => {
                let a1 = self.ka.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i];
                }
            }

            ([Some(s1), Some(s2)], [None, None]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let s2 = self.ka.get(s2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i] + s2[i];
                }
            }

            ([None, None], [Some(a1), Some(a2)]) => {
                let a1 = self.ka.get(a1.cast::<usize>()).assume();
                let a2 = self.ka.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] + a2[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i];
                }
            }

            ([Some(s1), None], [Some(a1), Some(a2)]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();
                let a2 = self.ka.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let s2 = self.ka.get(s2.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.ka.get(s1.cast::<usize>()).assume();
                let s2 = self.ka.get(s2.cast::<usize>()).assume();
                let a1 = self.ka.get(a1.cast::<usize>()).assume();
                let a2 = self.ka.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }

    /// Updates `acc` by adding and removing [`TIFeature`]s.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn accumulate_ti_in_place(
        &self,
        acc: &mut Aligned<[i16; N]>,
        sub: [Option<TIFeature>; 2],
        add: [Option<TIFeature>; 2],
    ) {
        match (sub, add) {
            ([Some(s1), None], [None, None]) => {
                let s1 = self.ti.get(s1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i];
                }
            }

            ([None, None], [Some(a1), None]) => {
                let a1 = self.ti.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i];
                }
            }

            ([Some(s1), Some(s2)], [None, None]) => {
                let s1 = self.ti.get(s1.cast::<usize>()).assume();
                let s2 = self.ti.get(s2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i] + s2[i];
                }
            }

            ([None, None], [Some(a1), Some(a2)]) => {
                let a1 = self.ti.get(a1.cast::<usize>()).assume();
                let a2 = self.ti.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] + a2[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.ti.get(s1.cast::<usize>()).assume();
                let a1 = self.ti.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i];
                }
            }

            ([Some(s1), None], [Some(a1), Some(a2)]) => {
                let s1 = self.ti.get(s1.cast::<usize>()).assume();
                let a1 = self.ti.get(a1.cast::<usize>()).assume();
                let a2 = self.ti.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.ti.get(s1.cast::<usize>()).assume();
                let s2 = self.ti.get(s2.cast::<usize>()).assume();
                let a1 = self.ti.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.ti.get(s1.cast::<usize>()).assume();
                let s2 = self.ti.get(s2.cast::<usize>()).assume();
                let a1 = self.ti.get(a1.cast::<usize>()).assume();
                let a2 = self.ti.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }
}
