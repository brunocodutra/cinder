use std::mem::transmute_copy;
use std::simd::{SimdElement, prelude::*};

/// Trait for [`Simd<_, N>` ] types that implement `halve`.
pub trait Halve {
    /// The output [`Simd<_, { N / 2 }>` ].
    type Output;

    /// Splits `self` in two halves.
    fn halve(self) -> [Self::Output; 2];
}

impl<T: SimdElement> Halve for Simd<T, 64> {
    type Output = Simd<T, 32>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn halve(self) -> [Self::Output; 2] {
        unsafe { transmute_copy::<Self, [Self::Output; 2]>(&self) }
    }
}

impl<T: SimdElement> Halve for Simd<T, 32> {
    type Output = Simd<T, 16>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn halve(self) -> [Self::Output; 2] {
        unsafe { transmute_copy::<Self, [Self::Output; 2]>(&self) }
    }
}

impl<T: SimdElement> Halve for Simd<T, 16> {
    type Output = Simd<T, 8>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn halve(self) -> [Self::Output; 2] {
        unsafe { transmute_copy::<Self, [Self::Output; 2]>(&self) }
    }
}

impl<T: SimdElement> Halve for Simd<T, 8> {
    type Output = Simd<T, 4>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn halve(self) -> [Self::Output; 2] {
        unsafe { transmute_copy::<Self, [Self::Output; 2]>(&self) }
    }
}
