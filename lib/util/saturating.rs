use crate::util::{Integer, Signed};
use derive_more::with_trait::{Debug, Display, Error};
use std::fmt::{self, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{cmp::Ordering, mem::size_of, num::Saturating as S, str::FromStr};

/// A saturating bounded integer.
#[derive(Debug, Default, Copy, Clone, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T, Self: Debug)))]
#[debug("Saturating({self})")]
#[debug(bounds(T: Integer<Repr: Signed>, T::Repr: Display))]
#[repr(transparent)]
pub struct Saturating<T>(T);

unsafe impl<T: Integer<Repr: Signed>> Integer for Saturating<T> {
    type Repr = T::Repr;
    const MIN: Self::Repr = T::MIN;
    const MAX: Self::Repr = T::MAX;
}

impl<T: Integer<Repr: Signed>> Eq for Saturating<T> where Self: PartialEq<Self> {}

impl<T: Integer<Repr: Signed>, U: Integer<Repr: Signed>> PartialEq<U> for Saturating<T> {
    #[inline(always)]
    fn eq(&self, other: &U) -> bool {
        if size_of::<T>() > size_of::<U>() {
            T::Repr::eq(&self.get(), &other.cast())
        } else {
            U::Repr::eq(&self.cast(), &other.get())
        }
    }
}

impl<T: Integer<Repr: Signed>> Ord for Saturating<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.get().cmp(&other.get())
    }
}

impl<T: Integer<Repr: Signed>, U: Integer<Repr: Signed>> PartialOrd<U> for Saturating<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &U) -> Option<Ordering> {
        if size_of::<T>() > size_of::<U>() {
            T::Repr::partial_cmp(&self.get(), &other.cast())
        } else {
            U::Repr::partial_cmp(&self.cast(), &other.get())
        }
    }
}

impl<T: Integer<Repr: Signed>> Neg for Saturating<T>
where
    S<T::Repr>: Neg<Output = S<T::Repr>>,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        S(self.get()).neg().0.saturate()
    }
}

impl<T: Integer<Repr: Signed>, U: Integer<Repr: Signed>> Add<U> for Saturating<T>
where
    S<T::Repr>: Add<Output = S<T::Repr>>,
    S<U::Repr>: Add<Output = S<U::Repr>>,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: U) -> Self::Output {
        if size_of::<T>() > size_of::<U>() {
            S::add(S(self.get()), S(rhs.cast())).0.saturate()
        } else {
            S::add(S(self.cast()), S(rhs.get())).0.saturate()
        }
    }
}

impl<T: Integer<Repr: Signed>, U: Integer<Repr: Signed>> Sub<U> for Saturating<T>
where
    S<T::Repr>: Sub<Output = S<T::Repr>>,
    S<U::Repr>: Sub<Output = S<U::Repr>>,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: U) -> Self::Output {
        if size_of::<T>() > size_of::<U>() {
            S::sub(S(self.get()), S(rhs.cast())).0.saturate()
        } else {
            S::sub(S(self.cast()), S(rhs.get())).0.saturate()
        }
    }
}

impl<T: Integer<Repr: Signed>, U: Integer<Repr: Signed>> Mul<U> for Saturating<T>
where
    S<T::Repr>: Mul<Output = S<T::Repr>>,
    S<U::Repr>: Mul<Output = S<U::Repr>>,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: U) -> Self::Output {
        if size_of::<T>() > size_of::<U>() {
            S::mul(S(self.get()), S(rhs.cast())).0.saturate()
        } else {
            S::mul(S(self.cast()), S(rhs.get())).0.saturate()
        }
    }
}

impl<T: Integer<Repr: Signed>, U: Integer<Repr: Signed>> Div<U> for Saturating<T>
where
    S<T::Repr>: Div<Output = S<T::Repr>>,
    S<U::Repr>: Div<Output = S<U::Repr>>,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: U) -> Self::Output {
        if size_of::<T>() > size_of::<U>() {
            S::div(S(self.get()), S(rhs.cast())).0.saturate()
        } else {
            S::div(S(self.cast()), S(rhs.get())).0.saturate()
        }
    }
}

impl<T: Integer<Repr: Signed>> Display for Saturating<T>
where
    T::Repr: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.get(), f)
    }
}

/// The reason why parsing [`Saturating`] failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("failed to parse saturating integer")]
pub struct ParseSaturatingIntegerError;

impl<T: Integer<Repr: Signed>> FromStr for Saturating<T>
where
    T::Repr: FromStr,
{
    type Err = ParseSaturatingIntegerError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<T::Repr>()
            .ok()
            .and_then(Integer::convert)
            .ok_or(ParseSaturatingIntegerError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
    #[cfg_attr(test, derive(test_strategy::Arbitrary))]
    #[repr(transparent)]
    struct Asymmetric(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

    unsafe impl Integer for Asymmetric {
        type Repr = i16;
        const MIN: Self::Repr = -89;
        const MAX: Self::Repr = 131;
    }

    #[proptest]
    fn comparison_coerces(a: Saturating<Asymmetric>, b: i8) {
        assert_eq!(a == b, a.get() == b.into());
        assert_eq!(a <= b, a.get() <= b.into());
    }

    #[proptest]
    fn negation_saturates(s: Saturating<Asymmetric>) {
        assert_eq!(-s, s.get().saturating_neg().saturate::<Asymmetric>());
    }

    #[proptest]
    fn addition_saturates(a: Saturating<Asymmetric>, b: Saturating<i8>) {
        let r: Asymmetric = i16::saturating_add(a.cast(), b.cast()).saturate();
        assert_eq!(a + b, r);

        let r: i8 = i16::saturating_add(b.cast(), a.cast()).saturate();
        assert_eq!(b + a, r);
    }

    #[proptest]
    fn subtraction_saturates(a: Saturating<Asymmetric>, b: Saturating<i8>) {
        let r: Asymmetric = i16::saturating_sub(a.cast(), b.cast()).saturate();
        assert_eq!(a - b, r);

        let r: i8 = i16::saturating_sub(b.cast(), a.cast()).saturate();
        assert_eq!(b - a, r);
    }

    #[proptest]
    fn multiplication_saturates(a: Saturating<Asymmetric>, b: Saturating<i8>) {
        let r: Asymmetric = i16::saturating_mul(a.cast(), b.cast()).saturate();
        assert_eq!(a * b, r);

        let r: i8 = i16::saturating_mul(b.cast(), a.cast()).saturate();
        assert_eq!(b * a, r);
    }

    #[proptest]
    fn division_saturates(
        #[filter(#a != 0)] a: Saturating<Asymmetric>,
        #[filter(#b != 0)] b: Saturating<i8>,
    ) {
        let r: Asymmetric = i16::saturating_div(a.cast(), b.cast()).saturate();
        assert_eq!(a / b, r);

        let r: i8 = i16::saturating_div(b.cast(), a.cast()).saturate();
        assert_eq!(b / a, r);
    }

    #[proptest]
    fn parsing_printed_saturating_integer_is_an_identity(a: Saturating<Asymmetric>) {
        assert_eq!(a.to_string().parse(), Ok(a));
    }

    #[proptest]
    fn parsing_saturating_integer_fails_for_numbers_too_small(
        #[strategy(..Saturating::<Asymmetric>::MIN)] n: i16,
    ) {
        assert_eq!(
            n.to_string().parse::<Saturating<Asymmetric>>(),
            Err(ParseSaturatingIntegerError)
        );
    }

    #[proptest]
    fn parsing_saturating_integer_fails_for_numbers_too_large(
        #[strategy(Saturating::<Asymmetric>::MAX + 1..)] n: i16,
    ) {
        assert_eq!(
            n.to_string().parse::<Saturating<Asymmetric>>(),
            Err(ParseSaturatingIntegerError)
        );
    }

    #[proptest]
    fn parsing_saturating_integer_fails_for_invalid_number(
        #[filter(#s.parse::<i16>().is_err())] s: String,
    ) {
        assert_eq!(
            s.to_string().parse::<Saturating<Asymmetric>>(),
            Err(ParseSaturatingIntegerError)
        );
    }
}
