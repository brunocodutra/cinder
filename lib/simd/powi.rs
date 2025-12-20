use std::ops::MulAssign;
use std::simd::{LaneCount, SimdElement, SupportedLaneCount, prelude::*};

/// Trait for [`Simd<_, _>` ] types that implement `powi`.
pub trait Powi {
    /// Raises `self` to the power of `E`.
    fn powi<const E: u32>(self) -> Self;
}

impl<T, const N: usize> Powi for Simd<T, N>
where
    T: SimdElement + From<i8>,
    LaneCount<N>: SupportedLaneCount,
    Self: MulAssign,
{
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn powi<const E: u32>(mut self) -> Self {
        let mut result = Self::splat(1.into());

        let mut exp = E;
        for _ in 0..32 - E.leading_zeros() {
            if exp & 1 == 1 {
                result *= self;
            }

            self *= self;
            exp >>= 1;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{array::UniformArrayStrategy, prelude::Strategy};
    use std::simd::StdFloat;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_i32(
        #[strategy(UniformArrayStrategy::new(-128i32..=127i32).prop_map(i32x4::from_array))]
        x: i32x4,
    ) {
        assert_eq!(x.powi::<1>(), x);
        assert_eq!(x.powi::<2>(), x * x);
        assert_eq!(x.powi::<3>(), x * x * x);
        assert_eq!(x.powi::<4>(), x * x * x * x);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn for_f32(
        #[strategy(UniformArrayStrategy::new(-128f32..=127f32).prop_map(f32x4::from_array))]
        x: f32x4,
    ) {
        let x = x.floor();
        assert_eq!(x.powi::<1>(), x);
        assert_eq!(x.powi::<2>(), x * x);
        assert_eq!(x.powi::<3>(), x * x * x);
        assert_eq!(x.powi::<4>(), x * x * x * x);
    }
}
