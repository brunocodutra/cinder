use std::simd::prelude::*;

/// Trait for [`Simd<_, _>` ] types that can search elements.
pub trait Find {
    type Bitmask;

    /// Returns a bitmask for the `needles` in `self`.
    fn find(self, needles: Self, count: usize) -> Self::Bitmask;
}

impl Find for u8x16 {
    type Bitmask = u16;

    #[inline(always)]
    #[cfg(target_feature = "sse4.2")]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn find(self, needles: Self, count: usize) -> Self::Bitmask {
        debug_assert!(count <= 16);

        unsafe {
            use std::arch::x86_64::*;
            _mm_extract_epi16::<0>(_mm_cmpestrm::<0>(
                needles.into(),
                count as i32,
                self.into(),
                16,
            )) as u16
        }
    }

    #[inline(always)]
    #[cfg(not(target_feature = "sse4.2"))]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn find(self, needles: Self, count: usize) -> Self::Bitmask {
        fallback(self, needles, count)
    }
}

#[allow(unused)]
#[inline(always)]
#[cfg_attr(feature = "no_panic", no_panic::no_panic)]
fn fallback(haystack: u8x16, needles: u8x16, count: usize) -> u16 {
    debug_assert!(count <= 16);

    let matches = (0..count).map(|i| haystack.simd_eq(u8x16::splat(needles[i])));
    matches.fold(0, |acc, m| acc | m.to_bitmask()) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u8x16(
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x16::from_array))] h: u8x16,
        #[strategy(UniformArrayStrategy::new(0u8..=255u8).prop_map(u8x16::from_array))] n: u8x16,
        #[strategy(..=16usize)] c: usize,
    ) {
        assert_eq!(h.find(n, c), fallback(h, n, c));
    }
}
