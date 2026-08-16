use crate::util::{Assume, Num};
use std::{array, simd::prelude::*};

/// Trait for [`Simd<_, _>` ] types that can shift left dynamically.
pub trait Shlv {
    /// Shifts left by `shift`.
    fn shlv(self, shift: Self) -> Self;
}

impl<const N: usize> Shlv for Simd<u16, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn shlv(self, shift: Self) -> Self {
        Simd::from_array(array::from_fn(|i| {
            self[i].checked_shl(shift[i].cast()).assume()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x64(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x64::from_array))]
        x: u16x64,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x64::from_array))] s: u16x64,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x32(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x32::from_array))]
        x: u16x32,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x32::from_array))] s: u16x32,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x16(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x16::from_array))]
        x: u16x16,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x16::from_array))] s: u16x16,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_u16x8(
        #[strategy(UniformArrayStrategy::new(0u16..=255u16).prop_map(u16x8::from_array))] x: u16x8,
        #[strategy(UniformArrayStrategy::new(0u16..8u16).prop_map(u16x8::from_array))] s: u16x8,
    ) {
        use crate::simd::Halve;
        let [x0, x1] = x.halve();
        let [s0, s1] = s.halve();
        assert_eq!(x.shlv(s).halve(), [x0.shlv(s0), x1.shlv(s1)]);
    }
}
