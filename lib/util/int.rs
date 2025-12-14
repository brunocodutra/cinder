use crate::util::{Assume, Float};
use bytemuck::{Pod, Zeroable, zeroed};
use std::fmt::{Binary, Debug, LowerHex, Octal, UpperHex};
use std::marker::Destruct;
use std::{cmp::Ordering, hash::Hash, hint::unreachable_unchecked, mem::transmute_copy};
use std::{num::*, ops::*};

/// Trait for types that can be represented by a int range of primitive integers.
///
/// # Safety
///
/// Must only be implemented for types that can be safely transmuted to and from [`Int::Repr`].
pub const unsafe trait Int: 'static + Send + Sync + Copy {
    /// The primitive integer representation.
    type Repr: [const] IntRepr;

    /// The minimum repr.
    const MIN: Self::Repr = <Self::Repr as Int>::MIN;

    /// The maximum repr.
    const MAX: Self::Repr = <Self::Repr as Int>::MAX;

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
    fn new(i: Self::Repr) -> Self {
        const { assert!(size_of::<Self>() == size_of::<Self::Repr>()) }
        const { assert!(align_of::<Self>() == align_of::<Self::Repr>()) }

        (Self::MIN..=Self::MAX).contains(&i).assume();
        unsafe { transmute_copy(&i) }
    }

    /// Casts to [`Int::Repr`].
    #[track_caller]
    #[inline(always)]
    fn get(self) -> Self::Repr {
        let repr = unsafe { transmute_copy(&self) };
        (Self::MIN..=Self::MAX).contains(&repr).assume();
        repr
    }

    /// Returns the sign of `self`.
    ///
    /// * `1` if `self > 0`
    /// * `0` if `self == 0`
    /// * `-1` if `self < 0`
    #[track_caller]
    #[inline(always)]
    fn signum(self) -> Self::Repr {
        self.get().cmp(&zero()).cast()
    }

    /// Casts to a [`IntRepr`].
    ///
    /// This is equivalent to the operator `as`.
    #[track_caller]
    #[inline(always)]
    fn cast<I: IntRepr>(self) -> I {
        self.get().cast()
    }

    /// Casts to [`Float`].
    #[track_caller]
    #[inline(always)]
    fn to_float<F: [const] Float>(self) -> F {
        self.get().to_float()
    }

    /// Converts to another [`Int`], if not out of range.
    #[track_caller]
    #[inline(always)]
    fn convert<I: [const] Int>(self) -> Option<I> {
        self.get().convert()
    }

    /// Converts to another [`Int`] with saturation.
    #[track_caller]
    #[inline(always)]
    fn saturate<I: [const] Int>(self) -> I {
        let min = I::MIN.convert().unwrap_or(Self::MIN);
        let max = I::MAX.convert().unwrap_or(Self::MAX);
        I::new(self.get().clamp(min, max).cast::<I::Repr>())
    }

    /// An iterator over all values in the range [`Int::MIN`]..=[`Int::MAX`].
    #[track_caller]
    #[inline(always)]
    fn iter() -> Ints<Self> {
        Ints(Self::MIN..=Self::MAX)
    }
}

#[derive(Debug)]
pub struct Ints<I: Int>(RangeInclusive<I::Repr>);

impl<I: Int> ExactSizeIterator for Ints<I>
where
    RangeInclusive<I::Repr>: ExactSizeIterator<Item = I::Repr>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<I: Int> Iterator for Ints<I>
where
    RangeInclusive<I::Repr>: ExactSizeIterator<Item = I::Repr>,
{
    type Item = I;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some(I::new(self.0.next()?))
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<I: Int> DoubleEndedIterator for Ints<I>
where
    RangeInclusive<I::Repr>: ExactSizeIterator<Item = I::Repr> + DoubleEndedIterator,
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(I::new(self.0.next_back()?))
    }
}

#[inline(always)]
pub const fn zero<U: Zeroable>() -> U {
    zeroed()
}

#[inline(always)]
pub const fn ones<U: Unsigned>(n: u32) -> U {
    match n {
        0 => zero(),
        n => unsafe { transmute_copy(&(u128::MAX >> (u128::BITS - n))) },
    }
}

/// Marker trait for primitive integers.
pub const trait IntRepr:
    [const] Int<Repr = Self>
    + Debug
    + Binary
    + Octal
    + LowerHex
    + UpperHex
    + [const] Destruct
    + [const] Default
    + [const] Eq
    + [const] PartialEq
    + [const] Ord
    + [const] PartialOrd
    + Hash
    + Zeroable
    + Pod
    + [const] Add<Output = Self>
    + [const] AddAssign
    + [const] Sub<Output = Self>
    + [const] SubAssign
    + [const] Mul<Output = Self>
    + [const] MulAssign
    + [const] Div<Output = Self>
    + [const] DivAssign
    + [const] BitAnd<Output = Self>
    + [const] BitAndAssign
    + [const] BitOr<Output = Self>
    + [const] BitOrAssign
    + [const] BitXor<Output = Self>
    + [const] BitXorAssign
    + [const] Shl<Output = Self>
    + [const] ShlAssign
    + [const] Shr<Output = Self>
    + [const] ShrAssign
    + [const] Not<Output = Self>
{
    /// This primitive's size in number of bits.
    const BITS: u32;
}

/// Marker trait for signed primitive integers.
pub const trait Signed: [const] IntRepr {}

/// Marker trait for unsigned primitive integers.
pub const trait Unsigned: [const] IntRepr {}

unsafe impl<I: [const] IntRepr> const Int for Saturating<I> {
    type Repr = I;
    const MIN: Self::Repr = I::MIN;
    const MAX: Self::Repr = I::MAX;
}

unsafe impl const Int for bool {
    type Repr = u8;
    const MIN: Self::Repr = 0x00;
    const MAX: Self::Repr = 0x01;
}

unsafe impl const Int for Ordering {
    type Repr = i8;
    const MIN: Self::Repr = Ordering::Less as _;
    const MAX: Self::Repr = Ordering::Greater as _;
}

macro_rules! impl_int_for_non_zero {
    ($nz: ty, $repr: ty) => {
        unsafe impl const Int for $nz {
            type Repr = $repr;
            const MIN: Self::Repr = Self::MIN.get();
            const MAX: Self::Repr = Self::MAX.get();
        }
    };
}

impl_int_for_non_zero!(NonZeroU8, u8);
impl_int_for_non_zero!(NonZeroU16, u16);
impl_int_for_non_zero!(NonZeroU32, u32);
impl_int_for_non_zero!(NonZeroU64, u64);
impl_int_for_non_zero!(NonZeroU128, u128);
impl_int_for_non_zero!(NonZeroUsize, usize);

macro_rules! impl_int_repr_for {
    ($i: ty) => {
        impl const IntRepr for $i {
            const BITS: u32 = <$i>::BITS;
        }

        unsafe impl const Int for $i {
            type Repr = $i;

            const MIN: Self::Repr = <$i>::MIN;
            const MAX: Self::Repr = <$i>::MAX;

            #[track_caller]
            #[inline(always)]
            fn cast<I: IntRepr>(self) -> I {
                if size_of::<I>() == size_of::<Self>() {
                    unsafe { transmute_copy(&self) }
                } else {
                    match size_of::<I>() {
                        1 => (self as u8).cast(),
                        2 => (self as u16).cast(),
                        4 => (self as u32).cast(),
                        8 => (self as u64).cast(),
                        16 => (self as u128).cast(),
                        _ => unsafe { unreachable_unchecked() },
                    }
                }
            }

            #[track_caller]
            #[inline(always)]
            fn to_float<F: Float>(self) -> F {
                match size_of::<F>() {
                    4 => unsafe { transmute_copy(&(self as f32)) },
                    8 => unsafe { transmute_copy(&(self as f64)) },
                    _ => unsafe { unreachable_unchecked() },
                }
            }

            #[track_caller]
            #[inline(always)]
            fn convert<I: [const] Int>(self) -> Option<I> {
                let i = self.cast();

                if (I::MIN..=I::MAX).contains(&i)
                    && i.cast::<Self>() == self
                    && (i < zero()) == (self < zero())
                {
                    Some(I::new(i))
                } else {
                    None
                }
            }
        }
    };
}

macro_rules! impl_signed_for {
    ($i: ty) => {
        impl_int_repr_for!($i);
        impl const Signed for $i {}
    };
}

impl_signed_for!(i8);
impl_signed_for!(i16);
impl_signed_for!(i32);
impl_signed_for!(i64);
impl_signed_for!(i128);
impl_signed_for!(isize);

macro_rules! impl_unsigned_for {
    ($i: ty) => {
        impl_int_repr_for!($i);
        impl const Unsigned for $i {}
    };
}

impl_unsigned_for!(u8);
impl_unsigned_for!(u16);
impl_unsigned_for!(u32);
impl_unsigned_for!(u64);
impl_unsigned_for!(u128);
impl_unsigned_for!(usize);

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

    unsafe impl const Int for Digit {
        type Repr = u16;
        const MIN: Self::Repr = Digit::One as _;
        const MAX: Self::Repr = Digit::Nine as _;
    }

    #[proptest]
    fn int_can_be_cast_from_repr(#[strategy(1u16..10)] i: u16) {
        assert_eq!(Digit::new(i).get(), i);
    }

    #[proptest]
    fn int_can_be_cast_to_repr(d: Digit) {
        assert_eq!(Digit::new(d.get()), d);
    }

    #[proptest]
    fn int_can_be_cast_to_primitive(d: Digit) {
        assert_eq!(d.cast::<i8>(), d.get() as i8);
    }

    #[proptest]
    fn int_can_be_cast_to_float(d: Digit) {
        assert_eq!(d.to_float::<f32>(), d.get() as f32);
    }

    #[proptest]
    fn int_can_be_converted_if_within_bounds(#[strategy(1i8..10)] i: i8) {
        assert_eq!(i.convert(), Some(Digit::new(i as u16)));
    }

    #[proptest]
    fn int_conversion_fails_if_smaller_than_min(#[strategy(..1i8)] i: i8) {
        assert_eq!(i.convert::<Digit>(), None);
    }

    #[proptest]
    fn int_conversion_fails_if_greater_than_max(#[strategy(10i8..)] i: i8) {
        assert_eq!(i.convert::<Digit>(), None);
    }

    #[proptest]
    fn int_can_be_converted_with_saturation(i: u8) {
        assert_eq!(i.saturate::<Digit>(), Digit::new(i.clamp(1, 9).into()));
    }

    #[test]
    fn int_can_be_iterated_in_order() {
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
    fn int_is_eq_by_repr(a: Digit, b: Digit) {
        assert_eq!(a == b, a.get() == b.get());
    }

    #[proptest]
    fn int_is_ord_by_repr(a: Digit, b: Digit) {
        assert_eq!(a < b, a.get() < b.get());
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
    fn primitive_can_be_converted_with_saturation(i: u16) {
        assert_eq!(i.saturate::<i8>(), i.min(i8::MAX as _) as i8);
        assert_eq!(i.saturate::<u32>(), u32::from(i));
    }
}
