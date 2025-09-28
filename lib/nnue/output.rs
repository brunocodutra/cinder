use crate::util::AlignTo64;
use bytemuck::Zeroable;

/// The output layer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Output<const N: usize> {
    #[cfg_attr(test, map(|b: i8| i32::from(b)))]
    pub bias: i32,
    #[cfg_attr(test, map(|vs: [i8; N]| AlignTo64(vs.map(i16::from))))]
    pub weight: AlignTo64<[i16; N]>,
}

impl<const N: usize> Output<N> {
    #[doc(hidden)]
    #[inline(always)]
    #[cfg(target_feature = "avx512f")]
    pub unsafe fn avx512(&self, us: &[i16; N], them: &[i16; N]) -> i32 {
        const { assert!(N.is_multiple_of(512)) }

        use crate::util::Assume;
        use std::{arch::x86_64::*, mem::transmute};

        #[inline(always)]
        unsafe fn dot(w: __m512i, x1: __m512i, x2: __m512i, y: __m512i) -> __m512i {
            unsafe {
                let x1 = _mm512_max_epi16(x1, _mm512_setzero_si512());
                let x1 = _mm512_min_epu16(x1, _mm512_set1_epi16(255));
                let x2 = _mm512_max_epi16(x2, _mm512_setzero_si512());
                let x2 = _mm512_min_epu16(x2, _mm512_set1_epi16(255));
                _mm512_add_epi32(y, _mm512_madd_epi16(x1, _mm512_mullo_epi16(x2, w)))
            }
        }

        unsafe {
            let mut y0 = _mm512_setzero_si512();
            let mut y1 = _mm512_setzero_si512();
            let mut y2 = _mm512_setzero_si512();
            let mut y3 = _mm512_setzero_si512();

            let (w1, w2) = self.weight.split_at_unchecked(N / 2);

            for (w, x) in [(w1, us), (w2, them)] {
                let (x1, x2) = x.split_at_unchecked(N / 2);
                let x1 = x1.as_chunks_unchecked::<256>();
                let x2 = x2.as_chunks_unchecked::<256>();
                let w = w.as_chunks_unchecked::<256>();

                for i in 0..N / 512 {
                    (x1.as_ptr() as usize).is_multiple_of(64).assume();
                    (x2.as_ptr() as usize).is_multiple_of(64).assume();
                    (w.as_ptr() as usize).is_multiple_of(64).assume();

                    let x1 = transmute::<&[i16; 256], &[__m512i; 8]>(&x1[i]);
                    let x2 = transmute::<&[i16; 256], &[__m512i; 8]>(&x2[i]);
                    let w = transmute::<&[i16; 256], &[__m512i; 8]>(&w[i]);

                    y0 = dot(w[0], x1[0], x2[0], y0);
                    y1 = dot(w[1], x1[1], x2[1], y1);
                    y2 = dot(w[2], x1[2], x2[2], y2);
                    y3 = dot(w[3], x1[3], x2[3], y3);
                    y0 = dot(w[4], x1[4], x2[4], y0);
                    y1 = dot(w[5], x1[5], x2[5], y1);
                    y2 = dot(w[6], x1[6], x2[6], y2);
                    y3 = dot(w[7], x1[7], x2[7], y3);
                }
            }

            let y01 = _mm512_add_epi32(y0, y1);
            let y23 = _mm512_add_epi32(y2, y3);
            let y = _mm512_add_epi32(y01, y23);

            self.bias + (_mm512_reduce_add_epi32(y) >> 9)
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn avx2(&self, us: &[i16; N], them: &[i16; N]) -> i32 {
        const { assert!(N.is_multiple_of(256)) }

        use crate::util::Assume;
        use std::{arch::x86_64::*, mem::transmute};

        #[inline(always)]
        unsafe fn dot(w: __m256i, x1: __m256i, x2: __m256i, y: __m256i) -> __m256i {
            unsafe {
                let x1 = _mm256_max_epi16(x1, _mm256_setzero_si256());
                let x1 = _mm256_min_epu16(x1, _mm256_set1_epi16(255));
                let x2 = _mm256_max_epi16(x2, _mm256_setzero_si256());
                let x2 = _mm256_min_epu16(x2, _mm256_set1_epi16(255));
                _mm256_add_epi32(y, _mm256_madd_epi16(x1, _mm256_mullo_epi16(x2, w)))
            }
        }

        unsafe {
            let mut y0 = _mm256_setzero_si256();
            let mut y1 = _mm256_setzero_si256();

            let (w1, w2) = self.weight.split_at_unchecked(N / 2);

            for (w, x) in [(w1, us), (w2, them)] {
                let (x1, x2) = x.split_at_unchecked(N / 2);
                let x1 = x1.as_chunks_unchecked::<128>();
                let x2 = x2.as_chunks_unchecked::<128>();
                let w = w.as_chunks_unchecked::<128>();

                for i in 0..N / 256 {
                    (x1.as_ptr() as usize).is_multiple_of(32).assume();
                    (x2.as_ptr() as usize).is_multiple_of(32).assume();
                    (w.as_ptr() as usize).is_multiple_of(32).assume();

                    let x1 = transmute::<&[i16; 128], &[__m256i; 8]>(&x1[i]);
                    let x2 = transmute::<&[i16; 128], &[__m256i; 8]>(&x2[i]);
                    let w = transmute::<&[i16; 128], &[__m256i; 8]>(&w[i]);

                    y0 = dot(w[0], x1[0], x2[0], y0);
                    y1 = dot(w[1], x1[1], x2[1], y1);
                    y0 = dot(w[2], x1[2], x2[2], y0);
                    y1 = dot(w[3], x1[3], x2[3], y1);
                    y0 = dot(w[4], x1[4], x2[4], y0);
                    y1 = dot(w[5], x1[5], x2[5], y1);
                    y0 = dot(w[6], x1[6], x2[6], y0);
                    y1 = dot(w[7], x1[7], x2[7], y1);
                }
            }

            let y = _mm256_add_epi32(y0, y1);

            // https://stackoverflow.com/a/60109639
            let r = _mm256_castsi256_si128(y);
            let s = _mm256_extracti128_si256(y, 1);
            let r = _mm_add_epi32(r, s);
            let s = _mm_unpackhi_epi64(r, r);
            let r = _mm_add_epi32(r, s);
            let s = _mm_shuffle_epi32(r, _MM_SHUFFLE(2, 3, 0, 1));
            let r = _mm_add_epi32(r, s);
            self.bias + (_mm_extract_epi32(r, 0) >> 9)
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    #[cfg(target_feature = "ssse3")]
    pub unsafe fn sse(&self, us: &[i16; N], them: &[i16; N]) -> i32 {
        const { assert!(N.is_multiple_of(128)) }

        use crate::util::Assume;
        use std::{arch::x86_64::*, mem::transmute};

        #[inline(always)]
        unsafe fn dot(w: __m128i, x1: __m128i, x2: __m128i, y: __m128i) -> __m128i {
            unsafe {
                let x1 = _mm_max_epi16(x1, _mm_setzero_si128());
                let x1 = _mm_min_epu16(x1, _mm_set1_epi16(255));
                let x2 = _mm_max_epi16(x2, _mm_setzero_si128());
                let x2 = _mm_min_epu16(x2, _mm_set1_epi16(255));
                _mm_add_epi32(y, _mm_madd_epi16(x1, _mm_mullo_epi16(x2, w)))
            }
        }

        unsafe {
            let mut y0 = _mm_setzero_si128();
            let mut y1 = _mm_setzero_si128();

            let (w1, w2) = self.weight.split_at_unchecked(N / 2);

            for (w, x) in [(w1, us), (w2, them)] {
                let (x1, x2) = x.split_at_unchecked(N / 2);
                let x1 = x1.as_chunks_unchecked::<64>();
                let x2 = x2.as_chunks_unchecked::<64>();
                let w = w.as_chunks_unchecked::<64>();

                for i in 0..N / 128 {
                    (x1.as_ptr() as usize).is_multiple_of(32).assume();
                    (x2.as_ptr() as usize).is_multiple_of(32).assume();
                    (w.as_ptr() as usize).is_multiple_of(32).assume();

                    let x1 = transmute::<&[i16; 64], &[__m128i; 8]>(&x1[i]);
                    let x2 = transmute::<&[i16; 64], &[__m128i; 8]>(&x2[i]);
                    let w = transmute::<&[i16; 64], &[__m128i; 8]>(&w[i]);

                    y0 = dot(w[0], x1[0], x2[0], y0);
                    y1 = dot(w[1], x1[1], x2[1], y1);
                    y0 = dot(w[2], x1[2], x2[2], y0);
                    y1 = dot(w[3], x1[3], x2[3], y1);
                    y0 = dot(w[4], x1[4], x2[4], y0);
                    y1 = dot(w[5], x1[5], x2[5], y1);
                    y0 = dot(w[6], x1[6], x2[6], y0);
                    y1 = dot(w[7], x1[7], x2[7], y1);
                }
            }

            let y = _mm_add_epi32(y0, y1);

            // https://stackoverflow.com/a/35270026
            let r = _mm_shuffle_epi32(y, _MM_SHUFFLE(1, 0, 3, 2));
            let s = _mm_add_epi32(r, y);
            let r = _mm_shufflelo_epi16(s, _MM_SHUFFLE(1, 0, 3, 2));
            let s = _mm_add_epi32(r, s);
            self.bias + (_mm_cvtsi128_si32(s) >> 9)
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    #[cfg(target_feature = "neon")]
    pub unsafe fn neon(&self, us: &[i16; N], them: &[i16; N]) -> i32 {
        const { assert!(N.is_multiple_of(128)) }

        use crate::util::Assume;
        use std::{arch::aarch64::*, mem::transmute};

        #[inline(always)]
        unsafe fn dot(w: int16x8_t, x1: int16x8_t, x2: int16x8_t, y: int32x4_t) -> int32x4_t {
            unsafe {
                let x1 = vmaxq_s16(x1, vdupq_n_s16(0));
                let x1 = vminq_s16(x1, vdupq_n_s16(255));
                let x2 = vmaxq_s16(x2, vdupq_n_s16(0));
                let x2 = vminq_s16(x2, vdupq_n_s16(255));
                let p = vmulq_s16(x2, w);
                vmlal_high_s16(vmlal_s16(y, vget_low_s16(x1), vget_low_s16(p)), x1, p)
            }
        }

        unsafe {
            let mut y0 = vdupq_n_s32(0);
            let mut y1 = vdupq_n_s32(0);
            let mut y2 = vdupq_n_s32(0);
            let mut y3 = vdupq_n_s32(0);

            let (w1, w2) = self.weight.split_at_unchecked(N / 2);

            for (w, x) in [(w1, us), (w2, them)] {
                let (x1, x2) = x.split_at_unchecked(N / 2);
                let x1 = x1.as_chunks_unchecked::<64>();
                let x2 = x2.as_chunks_unchecked::<64>();
                let w = w.as_chunks_unchecked::<64>();

                for i in 0..N / 128 {
                    (x1.as_ptr() as usize).is_multiple_of(32).assume();
                    (x2.as_ptr() as usize).is_multiple_of(32).assume();
                    (w.as_ptr() as usize).is_multiple_of(32).assume();

                    let x1 = transmute::<&[i16; 64], &[int16x8_t; 8]>(&x1[i]);
                    let x2 = transmute::<&[i16; 64], &[int16x8_t; 8]>(&x2[i]);
                    let w = transmute::<&[i16; 64], &[int16x8_t; 8]>(&w[i]);

                    y0 = dot(w[0], x1[0], x2[0], y0);
                    y1 = dot(w[1], x1[1], x2[1], y1);
                    y2 = dot(w[2], x1[2], x2[2], y2);
                    y3 = dot(w[3], x1[3], x2[3], y3);
                    y0 = dot(w[4], x1[4], x2[4], y0);
                    y1 = dot(w[5], x1[5], x2[5], y1);
                    y2 = dot(w[6], x1[6], x2[6], y2);
                    y3 = dot(w[7], x1[7], x2[7], y3);
                }
            }

            let y01 = vaddq_s32(y0, y1);
            let y23 = vaddq_s32(y2, y3);
            let y = vaddq_s32(y01, y23);
            self.bias + (vaddvq_s32(y) >> 9)
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn scalar(&self, us: &[i16; N], them: &[i16; N]) -> i32 {
        let mut y = 0;
        let (w1, w2) = unsafe { self.weight.split_at_unchecked(N / 2) };
        for (w, x) in [(w1, us), (w2, them)] {
            let (x1, x2) = unsafe { x.split_at_unchecked(N / 2) };
            for i in 0..N / 2 {
                y += w[i] as i32 * (x1[i] as i32).clamp(0, 255) * (x2[i] as i32).clamp(0, 255);
            }
        }

        self.bias + (y >> 9)
    }
}

impl<const N: usize> Output<N> {
    /// Transforms the accumulator.
    #[inline(always)]
    pub fn forward(&self, us: &AlignTo64<[i16; N]>, them: &AlignTo64<[i16; N]>) -> i32 {
        #[cfg(target_feature = "avx512f")]
        unsafe {
            self.avx512(us, them)
        }

        #[cfg(not(target_feature = "avx512f"))]
        #[cfg(target_feature = "avx2")]
        unsafe {
            self.avx2(us, them)
        }

        #[cfg(not(target_feature = "avx2"))]
        #[cfg(target_feature = "ssse3")]
        unsafe {
            self.sse(us, them)
        }

        #[cfg(target_feature = "neon")]
        unsafe {
            self.neon(us, them)
        }

        #[cfg(not(target_feature = "avx2"))]
        #[cfg(not(target_feature = "ssse3"))]
        #[cfg(not(target_feature = "neon"))]
        self.scalar(us, them)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[cfg(target_feature = "avx512f")]
    #[proptest]
    fn uses_avx512(o: Output<512>, i: AlignTo64<[[i16; 512]; 2]>) {
        assert_eq!(unsafe { o.avx512(&i[0], &i[1]) }, o.scalar(&i[0], &i[1]));
    }

    #[cfg(target_feature = "avx2")]
    #[proptest]
    fn uses_avx2(o: Output<256>, i: AlignTo64<[[i16; 256]; 2]>) {
        assert_eq!(unsafe { o.avx2(&i[0], &i[1]) }, o.scalar(&i[0], &i[1]));
    }

    #[cfg(target_feature = "ssse3")]
    #[proptest]
    fn uses_sse(o: Output<128>, i: AlignTo64<[[i16; 128]; 2]>) {
        assert_eq!(unsafe { o.sse(&i[0], &i[1]) }, o.scalar(&i[0], &i[1]));
    }

    #[cfg(target_feature = "neon")]
    #[proptest]
    fn uses_neon(o: Output<128>, i: AlignTo64<[[i16; 128]; 2]>) {
        assert_eq!(unsafe { o.neon(&i[0], &i[1]) }, o.scalar(&i[0], &i[1]));
    }
}
