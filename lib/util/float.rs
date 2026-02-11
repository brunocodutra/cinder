use crate::util::{Assume, Num, NumRepr};
use std::{hint::unreachable_unchecked, mem::transmute_copy, ops::*};

/// Trait for types that can be represented by a int range of primitive floats.
///
/// # Safety
///
/// * Must never be `NaN`.
/// * Must only be implemented for types that can be safely transmuted to and from [`Float::Repr`].
pub const unsafe trait Float: [const] Num
where
    Self::Repr: [const] FloatRepr,
{
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
pub const trait FloatRepr:
    [const] NumRepr
    + [const] Float<Repr = Self>
    + [const] PartialEq
    + [const] PartialOrd
    + [const] Add<Output = Self>
    + [const] AddAssign
    + [const] Sub<Output = Self>
    + [const] SubAssign
    + [const] Mul<Output = Self>
    + [const] MulAssign
    + [const] Div<Output = Self>
    + [const] DivAssign
{
}

macro_rules! impl_float_repr_for {
    ($f: ty) => {
        impl const NumRepr for $f {
            const IS_FLOAT: bool = true;
            const IS_SIGNED: bool = true;
        }

        unsafe impl const Num for $f {
            type Repr = $f;

            const MIN: Self::Repr = <$f>::NEG_INFINITY;
            const MAX: Self::Repr = <$f>::INFINITY;

            #[inline(always)]
            fn clip(self, min: Self, max: Self) -> Self {
                (min <= max).assume();
                <$f>::clamp(self, min, max)
            }

            #[inline(always)]
            fn cast<N: NumRepr>(self) -> N {
                if N::IS_FLOAT && size_of::<N>() == size_of::<Self>() {
                    unsafe { transmute_copy(&self) }
                } else if N::IS_FLOAT {
                    match size_of::<N>() {
                        4 => (self as f32).cast(),
                        8 => (self as f64).cast(),
                        _ => unsafe { unreachable_unchecked() },
                    }
                } else if N::IS_SIGNED {
                    match size_of::<N>() {
                        1 => (self as i8).cast(),
                        2 => (self as i16).cast(),
                        4 => (self as i32).cast(),
                        8 => (self as i64).cast(),
                        16 => (self as i128).cast(),
                        _ => unsafe { unreachable_unchecked() },
                    }
                } else {
                    match size_of::<N>() {
                        1 => (self as u8).cast(),
                        2 => (self as u16).cast(),
                        4 => (self as u32).cast(),
                        8 => (self as u64).cast(),
                        16 => (self as u128).cast(),
                        _ => unsafe { unreachable_unchecked() },
                    }
                }
            }

            #[inline(always)]
            fn convert<N: [const] Num<Repr: [const] NumRepr>>(self) -> Option<N> {
                let f = self.cast();

                #[expect(clippy::float_cmp)]
                if (N::MIN..=N::MAX).contains(&f) && f.cast::<Self>() == self {
                    Some(N::new(f))
                } else {
                    None
                }
            }

            #[inline(always)]
            fn saturate<N: [const] Num<Repr: [const] NumRepr>>(self) -> N {
                N::new(self.cast::<N::Repr>().clip(N::MIN, N::MAX))
            }
        }

        impl const FloatRepr for $f {}

        unsafe impl const Float for $f {
            #[inline(always)]
            fn lerp(self, a: Self, b: Self) -> Self {
                <$f>::mul_add(self, b - a, a)
            }
        }
    };
}

impl_float_repr_for!(f32);
impl_float_repr_for!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Arbitrary)]
    #[repr(transparent)]
    struct Unit(#[strategy(Self::MIN..=Self::MAX)] <Unit as Num>::Repr);

    unsafe impl const Num for Unit {
        type Repr = f64;
        const MIN: Self::Repr = -1.;
        const MAX: Self::Repr = 1.;
    }

    unsafe impl const Float for Unit {}

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn float_can_be_cast_from_repr(#[strategy(Unit::MIN..Unit::MAX)] f: f64) {
        assert_eq!(Unit::new(f).get(), f);
    }

    #[proptest]
    fn float_can_be_cast_to_repr(f: Unit) {
        assert_eq!(Unit::new(f.get()), f);
    }

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn float_can_be_cast_to_float(f: Unit) {
        assert_eq!(f.cast::<f32>(), f.get() as f32);
    }

    #[proptest]
    fn float_can_be_cast_to_int(f: Unit) {
        assert_eq!(f.cast::<i8>(), f.get() as i8);
    }

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn float_can_interpolate(#[filter(#f.get() >= 0.)] f: Unit, a: Unit, b: Unit) {
        assert_eq!(
            f.lerp(a, b).get(),
            f.get().mul_add(b.get() - a.get(), a.get())
        );
    }

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn float_is_eq_by_repr(a: Unit, b: Unit) {
        assert_eq!(a == b, a.get() == b.get());
    }

    #[proptest]
    fn float_is_ord_by_repr(a: Unit, b: Unit) {
        assert_eq!(a < b, a.get() < b.get());
    }

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn primitive_can_be_cast(f: f64) {
        assert_eq!(f.cast::<u32>(), f as u32);
        assert_eq!(f.cast::<i32>(), f as i32);
        assert_eq!(f.cast::<f32>(), f as f32);
    }

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn primitive_can_be_converted(#[strategy(i32::MIN as f64..=i32::MAX as f64)] f: f64) {
        assert_eq!(
            f.convert::<u32>(),
            (f == f as u32 as f64).then_some(f as u32)
        );

        assert_eq!(
            f.convert::<i32>(),
            (f == f as i32 as f64).then_some(f as i32)
        );

        assert_eq!(
            f.convert::<f32>(),
            (f == f as f32 as f64).then_some(f as f32)
        );

        assert_eq!(
            f.convert::<Unit>(),
            (Unit::MIN..=Unit::MAX).contains(&f).then(|| Unit::new(f))
        );
    }

    #[proptest]
    #[expect(clippy::float_cmp)]
    fn primitive_can_be_converted_with_saturation(f: f64) {
        assert_eq!(f.saturate::<u32>(), f as u32);
        assert_eq!(f.saturate::<i32>(), f as i32);
        assert_eq!(f.saturate::<f32>(), f as f32);

        assert_eq!(
            f.saturate::<Unit>(),
            Unit::new(f.clamp(Unit::MIN, Unit::MAX))
        );
    }
}
