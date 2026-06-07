use crate::simd::Aligned;
use std::simd::prelude::*;

/// Trait for [`Simd<_, _>` ] types that can broadcast a byte to its 8-byte lane.
pub trait Broadcast1x8 {
    /// Broadcasts a byte to its 8-byte lane.
    fn broadcast1x8(self) -> Self;
}

impl Broadcast1x8 for u8x64 {
    #[inline(always)]
    #[cfg(target_feature = "avx512bw")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        debug_assert_eq!(
            Aligned(self).as_ref::<[u8x8; 8]>().map(u8x8::reduce_sum),
            Aligned(self).as_ref::<[u8x8; 8]>().map(u8x8::reduce_max),
        );

        unsafe {
            use crate::simd::Shuffle;
            use std::arch::x86_64::*;

            #[rustfmt::skip]
            const INDICES: u8x64 = u8x64::from_array([
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
                0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
                0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
                0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,
                0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
                0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38,
            ]);

            const ZERO: u8x64 = Simd::splat(0);
            Shuffle::shuffle(_mm512_sad_epu8(self.into(), ZERO.into()).into(), INDICES)
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_feature = "avx512f", target_feature = "gfni")))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;

            let [x0, x1] = self.halve();
            transmute::<[u8x32; 2], Self>([x0.broadcast1x8(), x1.broadcast1x8()])
        }
    }
}

impl Broadcast1x8 for u8x32 {
    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        debug_assert_eq!(
            Aligned(self).as_ref::<[u8x8; 4]>().map(u8x8::reduce_sum),
            Aligned(self).as_ref::<[u8x8; 4]>().map(u8x8::reduce_max),
        );

        unsafe {
            use crate::simd::Shuffle;
            use std::arch::x86_64::*;

            #[rustfmt::skip]
            const INDICES: u8x32 = u8x32::from_array([
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
                0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
            ]);

            const ZERO: u8x32 = Simd::splat(0);
            Shuffle::shuffle(_mm256_sad_epu8(self.into(), ZERO.into()).into(), INDICES)
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        unsafe {
            use crate::simd::Halve;
            use std::mem::transmute;

            let [x0, x1] = self.halve();
            transmute::<[u8x16; 2], Self>([x0.broadcast1x8(), x1.broadcast1x8()])
        }
    }
}

impl Broadcast1x8 for u8x16 {
    #[inline(always)]
    #[cfg(target_feature = "sse2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        debug_assert_eq!(
            Aligned(self).as_ref::<[u8x8; 2]>().map(u8x8::reduce_sum),
            Aligned(self).as_ref::<[u8x8; 2]>().map(u8x8::reduce_max),
        );

        unsafe {
            use crate::simd::Shuffle;
            use std::arch::x86_64::*;

            #[rustfmt::skip]
            const INDICES: u8x16 = u8x16::from_array([
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
            ]);

            const ZERO: u8x16 = Simd::splat(0);
            Shuffle::shuffle(_mm_sad_epu8(self.into(), ZERO.into()).into(), INDICES)
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        use crate::simd::Halve;
        use std::mem::transmute;

        let [x0, x1] = self.halve();
        transmute::<[u8x8; 2], Self>([x0.broadcast1x8(), x1.broadcast1x8()])
    }
}

impl Broadcast1x8 for u8x8 {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn broadcast1x8(self) -> Self {
        fallback(self)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback(x: u8x8) -> u8x8 {
    let y = x.reduce_sum();
    debug_assert_eq!(y, x.reduce_max());
    u8x8::from_array([y, y, y, y, y, y, y, y])
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::array::UniformArrayStrategy;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x64(vs: [u8; 8], #[strategy(UniformArrayStrategy::new(..7usize))] is: [usize; 8]) {
        use crate::simd::Halve;

        let mut x = u8x64::splat(0);
        (0..8).for_each(|i| x[8 * i + is[i]] = vs[i]);

        let [x0, x1] = x.halve();

        assert_eq!(
            x.broadcast1x8().halve(),
            [x0.broadcast1x8(), x1.broadcast1x8()]
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x32(vs: [u8; 4], #[strategy(UniformArrayStrategy::new(..7usize))] is: [usize; 4]) {
        use crate::simd::Halve;

        let mut x = u8x32::splat(0);
        (0..4).for_each(|i| x[8 * i + is[i]] = vs[i]);

        let [x0, x1] = x.halve();

        assert_eq!(
            x.broadcast1x8().halve(),
            [x0.broadcast1x8(), x1.broadcast1x8()]
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x16(vs: [u8; 2], #[strategy(UniformArrayStrategy::new(..7usize))] is: [usize; 2]) {
        use crate::simd::Halve;

        let mut x = u8x16::splat(0);
        (0..2).for_each(|i| x[8 * i + is[i]] = vs[i]);

        let [x0, x1] = x.halve();

        assert_eq!(
            x.broadcast1x8().halve(),
            [x0.broadcast1x8(), x1.broadcast1x8()]
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x8(v: u8, #[strategy(..7usize)] i: usize) {
        let mut x = u8x8::splat(0);
        x[i] = v;

        assert_eq!(x.broadcast1x8(), fallback(x));
    }
}
