use crate::util::{Int, Signed};
use bytemuck::{Pod, Zeroable};
use derive_more::with_trait::{Debug, Display, Error};
use std::fmt::{self, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::{cmp::Ordering, mem::size_of, num::Saturating as S, str::FromStr};

/// A saturating bounded integer.
#[derive(Debug, Default, Copy, Clone, Hash, Zeroable, Pod)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T, Self: Debug)))]
#[debug("Bounded({self})")]
#[debug(bounds(T: Int<Repr: Signed>, T::Repr: Display))]
#[repr(transparent)]
pub struct Bounded<T>(T);

unsafe impl<T: Int<Repr: Signed>> Int for Bounded<T> {
    type Repr = T::Repr;
    const MIN: Self::Repr = T::MIN;
    const MAX: Self::Repr = T::MAX;
}

impl<T: Int<Repr: Signed>> Eq for Bounded<T> where Self: PartialEq<Self> {}

impl<T: Int<Repr: Signed>, U: Int<Repr: Signed>> PartialEq<U> for Bounded<T> {
    #[inline(always)]
    fn eq(&self, other: &U) -> bool {
        if size_of::<T>() > size_of::<U>() {
            T::Repr::eq(&self.get(), &other.cast())
        } else {
            U::Repr::eq(&self.cast(), &other.get())
        }
    }
}

impl<T: Int<Repr: Signed>> Ord for Bounded<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.get().cmp(&other.get())
    }
}

impl<T: Int<Repr: Signed>, U: Int<Repr: Signed>> PartialOrd<U> for Bounded<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &U) -> Option<Ordering> {
        if size_of::<T>() > size_of::<U>() {
            T::Repr::partial_cmp(&self.get(), &other.cast())
        } else {
            U::Repr::partial_cmp(&self.cast(), &other.get())
        }
    }
}

impl<T: Int<Repr: Signed>> Neg for Bounded<T>
where
    S<T::Repr>: Neg<Output = S<T::Repr>>,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        S(self.get()).neg().0.saturate()
    }
}

impl<T: Int<Repr: Signed>, U: Int<Repr: Signed>> Add<U> for Bounded<T>
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

impl<T: Int<Repr: Signed>, U> AddAssign<U> for Bounded<T>
where
    Self: Add<U, Output = Self>,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: U) {
        *self = *self + rhs
    }
}

impl<T: Int<Repr: Signed>, U: Int<Repr: Signed>> Sub<U> for Bounded<T>
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

impl<T: Int<Repr: Signed>, U> SubAssign<U> for Bounded<T>
where
    Self: Sub<U, Output = Self>,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: U) {
        *self = *self - rhs
    }
}

impl<T: Int<Repr: Signed>, U: Int<Repr: Signed>> Mul<U> for Bounded<T>
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

impl<T: Int<Repr: Signed>, U> MulAssign<U> for Bounded<T>
where
    Self: Mul<U, Output = Self>,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: U) {
        *self = *self * rhs
    }
}

impl<T: Int<Repr: Signed>, U: Int<Repr: Signed>> Div<U> for Bounded<T>
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

impl<T: Int<Repr: Signed>, U> DivAssign<U> for Bounded<T>
where
    Self: Div<U, Output = Self>,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: U) {
        *self = *self / rhs
    }
}

impl<T: Int<Repr: Signed>> Display for Bounded<T>
where
    T::Repr: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.get(), f)
    }
}

/// The reason why parsing [`Bounded`] failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("failed to parse bounded integer")]
pub struct ParseBoundedIntegerError;

impl<T: Int<Repr: Signed>> FromStr for Bounded<T>
where
    T::Repr: FromStr,
{
    type Err = ParseBoundedIntegerError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<T::Repr>()
            .ok()
            .and_then(Int::convert)
            .ok_or(ParseBoundedIntegerError)
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
    struct Asymmetric(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Int>::Repr);

    unsafe impl Int for Asymmetric {
        type Repr = i16;
        const MIN: Self::Repr = -89;
        const MAX: Self::Repr = 131;
    }

    #[proptest]
    fn comparison_coerces(a: Bounded<Asymmetric>, b: i8) {
        assert_eq!(a == b, a.get() == i16::from(b));
        assert_eq!(a <= b, a.get() <= i16::from(b));
    }

    #[proptest]
    fn negation_saturates(s: Bounded<Asymmetric>) {
        assert_eq!(-s, s.get().saturating_neg().saturate::<Asymmetric>());
    }

    #[proptest]
    fn addition_saturates(a: Bounded<Asymmetric>, b: Bounded<i8>) {
        let r: Asymmetric = i16::saturating_add(a.cast(), b.cast()).saturate();
        assert_eq!(a + b, r);

        let r: i8 = i16::saturating_add(b.cast(), a.cast()).saturate();
        assert_eq!(b + a, r);

        let mut c = a;
        c += b;
        assert_eq!(c, a + b);

        let mut c = b;
        c += a;
        assert_eq!(c, b + a);
    }

    #[proptest]
    fn subtraction_saturates(a: Bounded<Asymmetric>, b: Bounded<i8>) {
        let r: Asymmetric = i16::saturating_sub(a.cast(), b.cast()).saturate();
        assert_eq!(a - b, r);

        let r: i8 = i16::saturating_sub(b.cast(), a.cast()).saturate();
        assert_eq!(b - a, r);

        let mut c = a;
        c -= b;
        assert_eq!(c, a - b);

        let mut c = b;
        c -= a;
        assert_eq!(c, b - a);
    }

    #[proptest]
    fn multiplication_saturates(a: Bounded<Asymmetric>, b: Bounded<i8>) {
        let r: Asymmetric = i16::saturating_mul(a.cast(), b.cast()).saturate();
        assert_eq!(a * b, r);

        let r: i8 = i16::saturating_mul(b.cast(), a.cast()).saturate();
        assert_eq!(b * a, r);

        let mut c = a;
        c *= b;
        assert_eq!(c, a * b);

        let mut c = b;
        c *= a;
        assert_eq!(c, b * a);
    }

    #[proptest]
    fn division_saturates(
        #[filter(#a != 0)] a: Bounded<Asymmetric>,
        #[filter(#b != 0)] b: Bounded<i8>,
    ) {
        let r: Asymmetric = i16::saturating_div(a.cast(), b.cast()).saturate();
        assert_eq!(a / b, r);

        let r: i8 = i16::saturating_div(b.cast(), a.cast()).saturate();
        assert_eq!(b / a, r);

        let mut c = a;
        c /= b;
        assert_eq!(c, a / b);

        let mut c = b;
        c /= a;
        assert_eq!(c, b / a);
    }

    #[proptest]
    fn parsing_printed_bounded_integer_is_an_identity(a: Bounded<Asymmetric>) {
        assert_eq!(a.to_string().parse(), Ok(a));
    }

    #[proptest]
    fn parsing_bounded_integer_fails_for_numbers_too_small(
        #[strategy(..Bounded::<Asymmetric>::MIN)] n: i16,
    ) {
        assert_eq!(
            n.to_string().parse::<Bounded<Asymmetric>>(),
            Err(ParseBoundedIntegerError)
        );
    }

    #[proptest]
    fn parsing_bounded_integer_fails_for_numbers_too_large(
        #[strategy(Bounded::<Asymmetric>::MAX + 1..)] n: i16,
    ) {
        assert_eq!(
            n.to_string().parse::<Bounded<Asymmetric>>(),
            Err(ParseBoundedIntegerError)
        );
    }

    #[proptest]
    fn parsing_bounded_integer_fails_for_invalid_number(
        #[filter(#s.parse::<i16>().is_err())] s: String,
    ) {
        assert_eq!(
            s.to_string().parse::<Bounded<Asymmetric>>(),
            Err(ParseBoundedIntegerError)
        );
    }
}
