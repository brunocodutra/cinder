use crate::chess::{Move, Position};
use crate::util::{Assume, Primitive};
use derive_more::with_trait::Debug;
use std::sync::atomic::{AtomicI8, AtomicUsize, Ordering::Relaxed};

#[cfg(test)]
use proptest::prelude::*;

/// A trait for types that record statistics about [`Move`]s.
pub trait Statistics {
    /// The stat type.
    type Stat: Stat;

    /// Returns the accumulated [`Self::Stat`]s about a [`Move`] in a [`Position`].
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value;

    /// Updates [`Self::Stat`]s for a [`Move`] in a [`Position`].
    fn update(&self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value);
}

impl<T: Statistics> Statistics for &T {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        (*self).get(pos, m)
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        (*self).update(pos, m, delta)
    }
}

impl<T: Statistics> Statistics for Option<T> {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.as_ref()
            .map_or_else(Default::default, |g| g.get(pos, m))
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        if let Some(g) = self {
            g.update(pos, m, delta);
        }
    }
}

/// A trait for statistics counters.
pub trait Stat {
    /// The value type.
    type Value: Primitive;

    /// Returns the current [`Self::Value`].
    fn get(&self) -> Self::Value;

    /// Updates and returns the current [`Self::Value`].
    fn update(&self, delta: Self::Value);
}

impl<T: Stat> Stat for &T {
    type Value = T::Value;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        (*self).get()
    }

    #[inline(always)]
    fn update(&self, delta: Self::Value) {
        (*self).update(delta);
    }
}

impl<T: Stat> Stat for Option<T> {
    type Value = T::Value;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.as_ref().map_or_else(Default::default, Stat::get)
    }

    #[inline(always)]
    fn update(&self, delta: Self::Value) {
        if let Some(s) = self {
            s.update(delta);
        }
    }
}

/// A linear counter.
#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Counter(#[cfg_attr(test, strategy(any::<usize>().prop_map_into()))] AtomicUsize);

impl Stat for Counter {
    type Value = usize;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.0.load(Relaxed)
    }

    #[inline(always)]
    fn update(&self, delta: Self::Value) {
        self.0.fetch_add(delta, Relaxed);
    }
}

/// A saturating counter.
#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Graviton(#[cfg_attr(test, strategy(any::<i8>().prop_map_into()))] AtomicI8);

impl Stat for Graviton {
    type Value = i8;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.0.load(Relaxed)
    }

    #[inline(always)]
    fn update(&self, delta: Self::Value) {
        let delta = delta.max(-i8::MAX);
        let result = self.0.fetch_update(Relaxed, Relaxed, |h| {
            Some((delta as i16 - delta.abs() as i16 * h as i16 / 127 + h as i16) as i8)
        });

        result.assume();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn counter_accumulates_value(c: Counter, d: u8) {
        let prev = c.get();
        c.update(d as usize);
        assert_eq!(c.get(), prev + d as usize);
    }
}
