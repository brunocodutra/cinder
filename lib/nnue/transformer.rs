use crate::nnue::{Accumulator, KAFeature, Layer, PPFeature, TIFeature};
use crate::{simd::*, util::Num};
use bytemuck::Zeroable;
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;
use std::mem::transmute;

const K: usize = REGISTERS / 2;
const N: usize = Accumulator::LEN;

#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn load(chunk: &[i8; W4]) -> V4<i16> {
    V4::from_array(*chunk).cast()
}

/// The NNUE feature transformer.
#[derive(Debug, Zeroable)]
#[debug("Transformer")]
pub struct Transformer {
    pub bias: Aligned<[i16; N]>,
    pub ka: Aligned<[[i8; N]; KAFeature::LEN]>,
    pub ti: Aligned<[[i8; N]; TIFeature::LEN]>,
    pub pp: Aligned<[[i8; N]; PPFeature::LEN]>,
}

impl Transformer {
    /// Refreshes `accumulator`.
    #[inline(always)]
    pub fn refresh(&self, accumulator: &mut Aligned<[i16; N]>) {
        *accumulator = self.bias;
    }

    /// Updates `dst` by adding and removing features from `src`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    #[expect(clippy::too_many_arguments)]
    pub fn accumulate(
        &self,
        src: &Aligned<[i16; N]>,
        dst: &mut Aligned<[i16; N]>,
        kas: &[KAFeature],
        kaa: &[KAFeature],
        tis: &[TIFeature],
        tia: &[TIFeature],
        pps: &[PPFeature],
        ppa: &[PPFeature],
    ) {
        const { assert!(N.is_multiple_of(W4 * K)) }
        let src = src.cast_ref::<[[V4<i16>; K]; N / W4 / K]>();
        let dst = dst.cast_mut::<[[V4<i16>; K]; N / W4 / K]>();

        let kal = kas.len().max(kaa.len());
        let til = tis.len().max(tia.len());
        let ppl = pps.len().max(ppa.len());

        for i in 0..N / W4 / K {
            let mut regs = src[i];

            for j in 0..kal {
                match (kas.get(j), kaa.get(j)) {
                    (Some(s), Some(a)) => unsafe {
                        let s = self.ka.get_unchecked(s.cast::<usize>());
                        let a = self.ka.get_unchecked(a.cast::<usize>());

                        let s = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(s)[i];
                        let a = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(a)[i];

                        for k in 0..K {
                            regs[k] += load(&a[k]) - load(&s[k]);
                        }
                    },

                    (Some(s), None) => unsafe {
                        let s = self.ka.get_unchecked(s.cast::<usize>());
                        let s = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(s)[i];

                        for k in 0..K {
                            regs[k] -= load(&s[k]);
                        }
                    },

                    (None, Some(a)) => unsafe {
                        let a = self.ka.get_unchecked(a.cast::<usize>());
                        let a = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(a)[i];

                        for k in 0..K {
                            regs[k] += load(&a[k]);
                        }
                    },

                    (None, None) => unsafe { unreachable_unchecked() },
                }
            }

            for j in 0..til {
                match (tis.get(j), tia.get(j)) {
                    (Some(s), Some(a)) => unsafe {
                        let s = self.ti.get_unchecked(s.cast::<usize>());
                        let a = self.ti.get_unchecked(a.cast::<usize>());

                        let s = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(s)[i];
                        let a = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(a)[i];

                        for k in 0..K {
                            regs[k] += load(&a[k]) - load(&s[k]);
                        }
                    },

                    (Some(s), None) => unsafe {
                        let s = self.ti.get_unchecked(s.cast::<usize>());
                        let s = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(s)[i];

                        for k in 0..K {
                            regs[k] -= load(&s[k]);
                        }
                    },

                    (None, Some(a)) => unsafe {
                        let a = self.ti.get_unchecked(a.cast::<usize>());
                        let a = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(a)[i];

                        for k in 0..K {
                            regs[k] += load(&a[k]);
                        }
                    },

                    (None, None) => unsafe { unreachable_unchecked() },
                }
            }

            for j in 0..ppl {
                match (pps.get(j), ppa.get(j)) {
                    (Some(s), Some(a)) => unsafe {
                        let s = self.pp.get_unchecked(s.cast::<usize>());
                        let a = self.pp.get_unchecked(a.cast::<usize>());

                        let s = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(s)[i];
                        let a = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(a)[i];

                        for k in 0..K {
                            regs[k] += load(&a[k]) - load(&s[k]);
                        }
                    },

                    (Some(s), None) => unsafe {
                        let s = self.pp.get_unchecked(s.cast::<usize>());
                        let s = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(s)[i];

                        for k in 0..K {
                            regs[k] -= load(&s[k]);
                        }
                    },

                    (None, Some(a)) => unsafe {
                        let a = self.pp.get_unchecked(a.cast::<usize>());
                        let a = &transmute::<&[i8; N], &[[[i8; W4]; K]; N / W4 / K]>(a)[i];

                        for k in 0..K {
                            regs[k] += load(&a[k]);
                        }
                    },

                    (None, None) => unsafe { unreachable_unchecked() },
                }
            }

            dst[i] = regs;
        }
    }
}
