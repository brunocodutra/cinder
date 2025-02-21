use crate::util::Assume;
use std::{hint::unreachable_unchecked, mem::transmute_copy, num::*, ops::*};

/// Trait for types that can be represented by a contiguous range of primitive integers.
///
/// # Safety
///
/// Must only be implemented for types that can be safely transmuted to and from [`Integer::Repr`].
pub unsafe trait Integer: Copy {
    /// The equivalent primitive integer type.
    type Repr: Primitive;

    /// The minimum repr.
    const MIN: Self::Repr;

    /// The maximum repr.
    const MAX: Self::Repr;

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

    /// Casts from [`Integer::Repr`].
    #[track_caller]
    #[inline(always)]
    fn new(i: Self::Repr) -> Self {
        (Self::MIN..=Self::MAX).contains(&i).assume();
        unsafe { transmute_copy(&i) }
    }

    /// Casts to [`Integer::Repr`].
    #[inline(always)]
    fn get(self) -> Self::Repr {
        let repr = unsafe { transmute_copy(&self) };
        (Self::MIN..=Self::MAX).contains(&repr).assume();
        repr
    }

    /// Casts to a [`Primitive`].
    ///
    /// This is equivalent to the operator `as`.
    #[inline(always)]
    fn cast<I: Primitive>(self) -> I {
        self.get().cast()
    }

    /// Converts to another [`Integer`] if possible without data loss.
    #[inline(always)]
    fn convert<I: Integer>(self) -> Option<I> {
        self.get().convert()
    }

    /// Converts to another [`Integer`] with saturation.
    #[inline(always)]
    fn saturate<I: Integer>(self) -> I {
        let min = I::MIN.convert().unwrap_or(Self::MIN);
        let max = I::MAX.convert().unwrap_or(Self::MAX);
        I::new(self.get().clamp(min, max).cast::<I::Repr>())
    }

    /// An iterator over all values in the range [`Integer::MIN`]..=[`Integer::MAX`].
    #[inline(always)]
    fn iter() -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator
    where
        RangeInclusive<Self::Repr>: ExactSizeIterator<Item = Self::Repr> + DoubleEndedIterator,
    {
        (Self::MIN..=Self::MAX).map(Self::new)
    }
}

/// Trait for primitive integer types.
pub trait Primitive:
    Integer<Repr = Self>
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + BitXor<Output = Self>
    + BitXorAssign
    + Shl<Output = Self>
    + ShlAssign
    + Shr<Output = Self>
    + ShrAssign
    + Not<Output = Self>
{
    /// The bit width.
    const BITS: u32;

    /// The constant `0`.
    #[inline(always)]
    fn zero() -> Self {
        Self::ones(0)
    }

    /// A value with `n` trailing `1`s.
    fn ones(n: u32) -> Self;
}

/// Marker trait for signed primitive integers.
pub trait Signed: Primitive {}

/// Marker trait for unsigned primitive integers.
pub trait Unsigned: Primitive {}

unsafe impl<I: Primitive> Integer for Saturating<I> {
    type Repr = I;
    const MIN: Self::Repr = I::MIN;
    const MAX: Self::Repr = I::MAX;
}

macro_rules! impl_integer_for_non_zero {
    ($nz: ty, $repr: ty) => {
        unsafe impl Integer for $nz {
            type Repr = $repr;
            const MIN: Self::Repr = Self::MIN.get();
            const MAX: Self::Repr = Self::MAX.get();
        }
    };
}

impl_integer_for_non_zero!(NonZeroU8, u8);
impl_integer_for_non_zero!(NonZeroU16, u16);
impl_integer_for_non_zero!(NonZeroU32, u32);
impl_integer_for_non_zero!(NonZeroU64, u64);
impl_integer_for_non_zero!(NonZeroU128, u128);
impl_integer_for_non_zero!(NonZeroUsize, usize);

macro_rules! impl_primitive_for {
    ($i: ty, $m: ty) => {
        impl $m for $i {}

        impl Primitive for $i {
            const BITS: u32 = <$i>::BITS;

            #[inline(always)]
            fn ones(n: u32) -> Self {
                match n {
                    0 => 0,
                    n => Self::MAX >> (Self::BITS - n),
                }
            }
        }

        unsafe impl Integer for $i {
            type Repr = $i;

            const MIN: Self::Repr = <$i>::MIN;
            const MAX: Self::Repr = <$i>::MAX;

            #[inline(always)]
            fn cast<I: Primitive>(self) -> I {
                if I::BITS <= Self::BITS {
                    unsafe { transmute_copy(&self) }
                } else {
                    match I::BITS {
                        16 => (self as i16).cast(),
                        32 => (self as i32).cast(),
                        64 => (self as i64).cast(),
                        128 => (self as i128).cast(),
                        _ => unsafe { unreachable_unchecked() },
                    }
                }
            }

            #[inline(always)]
            fn convert<I: Integer>(self) -> Option<I> {
                let i = self.cast();

                if (I::MIN..=I::MAX).contains(&i)
                    && i.cast::<Self>() == self
                    && (i < I::Repr::zero()) == (self < Self::zero())
                {
                    Some(I::new(i))
                } else {
                    None
                }
            }
        }
    };
}

impl_primitive_for!(i8, Signed);
impl_primitive_for!(i16, Signed);
impl_primitive_for!(i32, Signed);
impl_primitive_for!(i64, Signed);
impl_primitive_for!(i128, Signed);
impl_primitive_for!(isize, Signed);

impl_primitive_for!(u8, Unsigned);
impl_primitive_for!(u16, Unsigned);
impl_primitive_for!(u32, Unsigned);
impl_primitive_for!(u64, Unsigned);
impl_primitive_for!(u128, Unsigned);
impl_primitive_for!(usize, Unsigned);

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Arbitrary)]
    #[repr(u16)]
    enum Digit {
        One = 1,
        Two,
        Three,
        Four,
        Five,
        Six,
        Seven,
        Eight,
        Nine,
    }

    unsafe impl Integer for Digit {
        type Repr = u16;
        const MIN: Self::Repr = Digit::One as _;
        const MAX: Self::Repr = Digit::Nine as _;
    }

    #[proptest]
    fn integer_can_be_cast_from_repr(#[strategy(1u16..10)] i: u16) {
        assert_eq!(Digit::new(i).get(), i);
    }

    #[proptest]
    fn integer_can_be_cast_to_repr(d: Digit) {
        assert_eq!(Digit::new(d.get()), d);
    }

    #[proptest]
    fn integer_can_be_cast_to_primitive(d: Digit) {
        assert_eq!(d.cast::<i8>(), d.get() as i8);
    }

    #[proptest]
    fn integer_can_be_converted_to_another_integer_within_bounds(#[strategy(1i8..10)] i: i8) {
        assert_eq!(i.convert(), Some(Digit::new(i as u16)));
    }

    #[proptest]
    fn integer_conversion_fails_if_smaller_than_min(#[strategy(..1i8)] i: i8) {
        assert_eq!(i.convert::<Digit>(), None);
    }

    #[proptest]
    fn integer_conversion_fails_if_greater_than_max(#[strategy(10i8..)] i: i8) {
        assert_eq!(i.convert::<Digit>(), None);
    }

    #[proptest]
    fn integer_can_be_converted_to_another_integer_with_saturation(i: u8) {
        assert_eq!(i.saturate::<Digit>(), Digit::new(i.clamp(1, 9).into()));
    }

    #[test]
    fn integer_can_be_iterated_in_order() {
        assert_eq!(
            Vec::from_iter(Digit::iter()),
            vec![
                Digit::One,
                Digit::Two,
                Digit::Three,
                Digit::Four,
                Digit::Five,
                Digit::Six,
                Digit::Seven,
                Digit::Eight,
                Digit::Nine,
            ],
        );
    }

    #[proptest]
    fn integer_is_eq_by_repr(a: Digit, b: Digit) {
        assert_eq!(a == b, a.get() == b.get());
    }

    #[proptest]
    fn integer_is_ord_by_repr(a: Digit, b: Digit) {
        assert_eq!(a < b, a.get() < b.get());
    }

    #[proptest]
    fn primitive_can_be_constructed_with_trailing_ones(#[strategy(..=64u32)] n: u32) {
        assert_eq!(u64::ones(n).trailing_ones(), n);
    }

    #[proptest]
    fn primitive_can_be_cast(i: i16) {
        assert_eq!(i.cast::<u8>(), i as u8);
        assert_eq!(i.cast::<i8>(), i as i8);

        assert_eq!(i.cast::<u32>(), i as u32);
        assert_eq!(i.cast::<i32>(), i as i32);

        assert_eq!(i.cast::<u8>().cast::<i32>(), i as u8 as i32);
        assert_eq!(i.cast::<i8>().cast::<u32>(), i as i8 as u32);

        assert_eq!(i.cast::<u32>().cast::<i8>(), i as u32 as i8);
        assert_eq!(i.cast::<i32>().cast::<u8>(), i as i32 as u8);
    }

    #[proptest]
    fn primitive_can_be_converted(#[strategy(256u16..)] i: u16) {
        assert_eq!(i.convert::<u8>(), None);
        assert_eq!(i.convert::<i8>(), None);

        assert_eq!(i.convert::<u32>(), Some(i.into()));
        assert_eq!(i.convert::<i32>(), Some(i.into()));
    }

    #[proptest]
    fn integer_can_be_converted_to_another_primitive_with_saturation(i: u16) {
        assert_eq!(i.saturate::<i8>(), i.min(i8::MAX as _) as _);
        assert_eq!(i.saturate::<u32>(), i.into());
    }
}
