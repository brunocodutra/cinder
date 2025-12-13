use crate::util::{Assume, Int, Signed};
use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, hint::unreachable_unchecked, mem::transmute_copy, ops::*};

/// Trait for finite floating point primitive types.
///
/// # Safety
///
/// Must only be implemented for types that can be safely transmuted to and from [`Float::Repr`].
pub unsafe trait Float: 'static + Send + Sync + Copy {
    /// The primitive float representation.
    type Repr: FloatRepr;

    /// The minimum repr.
    const MIN: Self::Repr = <Self::Repr as Float>::MIN;

    /// The maximum repr.
    const MAX: Self::Repr = <Self::Repr as Float>::MAX;

    /// The minimum value.
    #[inline(always)]
    fn lower() -> Self {
        Self::new(Self::MIN)
    }

    /// The maximum value.
    #[inline(always)]
    fn upper() -> Self {
        Self::new(Self::MAX)
    }

    /// Casts from [`Int::Repr`].
    #[track_caller]
    #[inline(always)]
    fn new(f: Self::Repr) -> Self {
        const { assert!(size_of::<Self>() == size_of::<Self::Repr>()) }
        const { assert!(align_of::<Self>() == align_of::<Self::Repr>()) }

        (Self::MIN..=Self::MAX).contains(&f).assume();
        unsafe { transmute_copy(&f) }
    }

    /// Casts to [`Int::Repr`].
    #[track_caller]
    #[inline(always)]
    fn get(self) -> Self::Repr {
        let repr = unsafe { transmute_copy(&self) };
        (Self::MIN..=Self::MAX).contains(&repr).assume();
        repr
    }

    /// Converts to [`Int`] with saturation.
    #[track_caller]
    #[inline(always)]
    fn to_int<I: Int<Repr: Signed>>(self) -> I {
        self.get().to_int()
    }

    /// Linearly interpolates between `a` and `b`.
    ///
    /// When `self` is 0, returns `a`. When `self` is 1, returns `b`.
    #[track_caller]
    #[inline(always)]
    fn lerp(self, a: Self, b: Self) -> Self {
        Self::new(self.get().lerp(a.get(), b.get()))
    }
}

/// Marker trait for primitive floats.
pub trait FloatRepr:
    Float<Repr = Self>
    + Debug
    + Default
    + PartialEq
    + PartialOrd
    + Zeroable
    + Pod
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
{
}

impl FloatRepr for f32 {}
impl FloatRepr for f64 {}

unsafe impl Float for f32 {
    type Repr = f32;

    const MIN: Self::Repr = f32::MIN;
    const MAX: Self::Repr = f32::MAX;

    #[inline(always)]
    fn to_int<I: Int<Repr: Signed>>(self) -> I {
        self.is_finite().assume();

        match size_of::<I>() {
            1 => (self as i8).saturate(),
            2 => (self as i16).saturate(),
            4 => (self as i32).saturate(),
            8 => (self as i64).saturate(),
            16 => (self as i128).saturate(),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn lerp(self, a: Self, b: Self) -> Self {
        self.mul_add(b - a, a)
    }
}

unsafe impl Float for f64 {
    type Repr = f64;

    const MIN: Self::Repr = f64::MIN;
    const MAX: Self::Repr = f64::MAX;

    #[inline(always)]
    fn to_int<I: Int<Repr: Signed>>(self) -> I {
        self.is_finite().assume();

        match size_of::<I>() {
            1 => (self as i8).saturate(),
            2 => (self as i16).saturate(),
            4 => (self as i32).saturate(),
            8 => (self as i64).saturate(),
            16 => (self as i128).saturate(),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn lerp(self, a: Self, b: Self) -> Self {
        self.mul_add(b - a, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Arbitrary)]
    #[repr(transparent)]
    struct Unit(#[strategy(Self::MIN..=Self::MAX)] <Unit as Float>::Repr);

    unsafe impl Float for Unit {
        type Repr = f64;
        const MIN: Self::Repr = -1.;
        const MAX: Self::Repr = 1.;
    }

    #[proptest]
    fn float_can_be_cast_from_repr(#[strategy(Unit::MIN..Unit::MAX)] f: f64) {
        assert_eq!(Unit::new(f).get(), f);
    }

    #[proptest]
    fn float_can_be_cast_to_repr(f: Unit) {
        assert_eq!(Unit::new(f.get()), f);
    }

    #[proptest]
    fn float_can_be_cast_to_int(f: Unit) {
        assert_eq!(f.to_int::<i8>(), f.get() as i8);
    }

    #[proptest]
    fn float_can_interpolate(#[filter(#f.get() >= 0.)] f: Unit, a: Unit, b: Unit) {
        assert_eq!(
            f.lerp(a, b).get(),
            f.get().mul_add(b.get() - a.get(), a.get())
        );
    }

    #[proptest]
    fn float_is_eq_by_repr(a: Unit, b: Unit) {
        assert_eq!(a == b, a.get() == b.get());
    }

    #[proptest]
    fn float_is_ord_by_repr(a: Unit, b: Unit) {
        assert_eq!(a < b, a.get() < b.get());
    }
}
