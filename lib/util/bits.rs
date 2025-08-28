use crate::util::{Integer, Unsigned};
use bytemuck::{NoUninit, Zeroable};
use derive_more::with_trait::{Debug, *};
use std::mem::transmute_copy;
use std::ops::{Bound, Not, RangeBounds};

#[cfg(test)]
use proptest::prelude::*;

#[cfg(test)]
use std::ops::RangeInclusive;

/// A fixed width collection of bits.
#[derive(
    Debug,
    Display,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Zeroable,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    BitXor,
    BitXorAssign,
)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T, T: Unsigned, Self: Debug, RangeInclusive<T>: Strategy<Value = T>)))]
#[debug("Bits({_0:b})")]
#[display("{_0:b}")]
#[repr(transparent)]
pub struct Bits<T, const W: u32>(#[cfg_attr(test, strategy(T::zero()..=T::ones(W)))] T);

/// Workaround to https://github.com/rust-lang/rfcs/pull/3762.
pub const fn bits<U: Integer<Repr: Unsigned>>(i: u128) -> U {
    unsafe { transmute_copy(&i) }
}

unsafe impl<T: Unsigned + 'static, const W: u32> NoUninit for Bits<T, W> {}

unsafe impl<T: Unsigned + 'static, const W: u32> Integer for Bits<T, W> {
    type Repr = T;
    const MIN: Self::Repr = bits(0);
    const MAX: Self::Repr = bits(u128::MAX >> (u128::BITS - W));
}

impl<T: Unsigned, const W: u32> Bits<T, W> {
    /// The bit width.
    pub const BITS: u32 = const {
        assert!(size_of::<T>() * 8 >= W as usize);
        W
    };

    /// Whether this is a superset of `bits`.
    #[inline(always)]
    pub fn contains(&self, bits: &Self) -> bool {
        *self & *bits == *bits
    }

    /// Returns a slice of bits.
    #[inline(always)]
    pub fn slice<R: RangeBounds<u32>>(&self, r: R) -> Self {
        let a = match r.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };

        let b = match r.end_bound() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => W,
        };

        Bits::new((self.get() & T::ones(b)) >> a.cast())
    }

    /// Shifts bits into the collection.
    #[inline(always)]
    pub fn push<U: Unsigned, const N: u32>(&mut self, bits: Bits<U, N>) {
        *self = Bits::new((self.get() << N.cast()) & T::ones(W) ^ bits.cast());
    }

    /// Shifts bits out of the collection.
    #[inline(always)]
    pub fn pop<U: Unsigned, const N: u32>(&mut self) -> Bits<U, N> {
        let bits = Bits::new(self.cast::<U>() & U::ones(N));
        *self = Bits::new(self.get() >> N.cast());
        bits
    }
}

impl<T: Unsigned, const W: u32> Default for Bits<T, W> {
    #[inline(always)]
    fn default() -> Self {
        Bits::new(T::zero())
    }
}

impl<T: Unsigned, const W: u32> Not for Bits<T, W> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        self ^ Bits::new(T::ones(W))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Primitive;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn can_be_constructed_from_raw_collection_of_bits(n: u8) {
        assert_eq!(Bits::<_, 8>::new(n), Bits(n));
    }

    #[proptest]
    fn get_returns_raw_collection_of_bits(b: Bits<u8, 5>) {
        assert_eq!(b.get(), b.0);
    }

    #[proptest]
    fn slice_retrieves_range_of_bits(b: Bits<u8, 8>, #[strategy(..8u32)] i: u32) {
        assert_eq!(b.slice(0..), b.slice(..));
        assert_eq!(b.slice(..8), b.slice(..));
        assert_eq!(b.slice(0..8), b.slice(..));

        assert_eq!(b.slice(i..i), Bits::default());
        assert_eq!(b.slice(..=i), b.slice(..i + 1));
    }

    #[proptest]
    #[should_panic]
    fn slice_panics_if_index_is_out_of_range(b: Bits<u64, 48>, #[strategy(48u32..)] i: u32) {
        b.slice(i..i);
    }

    #[proptest]
    fn always_contains_prefix(b: Bits<u8, 8>, #[strategy(..=8u32)] i: u32) {
        assert!(b.contains(&b.slice(..i)));
    }

    #[proptest]
    fn push_shifts_bits_into_collection(a: Bits<u8, 3>, b: Bits<u16, 9>, c: Bits<u32, 27>) {
        let mut bits = Bits::<u64, 39>::default();

        bits.push(a);
        bits.push(b);
        bits.push(c);

        assert_eq!(bits.slice(36..).get(), u64::from(a.get()));
        assert_eq!(bits.slice(27..36).get(), u64::from(b.get()));
        assert_eq!(bits.slice(..27).get(), u64::from(c.get()));
    }

    #[proptest]
    #[should_panic]
    fn push_panics_on_overflow(mut a: Bits<u8, 3>, b: Bits<u16, 9>) {
        a.push(b);
    }

    #[proptest]
    fn pop_removes_pushed_bits(a: Bits<u8, 3>, b: Bits<u16, 9>, c: Bits<u32, 27>) {
        let mut bits = Bits::<u64, 39>::default();

        bits.push(a);
        bits.push(b);
        bits.push(c);

        assert_eq!(bits.pop(), c);
        assert_eq!(bits.pop(), b);
        assert_eq!(bits.pop(), a);
    }

    #[proptest]
    #[should_panic]
    fn pop_panics_on_underflow(mut a: Bits<u8, 3>) {
        a.pop::<u16, 9>();
    }

    #[proptest]
    fn not_inverts_bits(b: Bits<u8, 5>) {
        assert_ne!((!b).get(), !b.get());
        assert_eq!((!b).get(), !b.get() & u8::ones(5));
    }
}
