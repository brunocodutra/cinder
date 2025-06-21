use crate::chess::{Move, Position};
use crate::util::{Assume, Integer, Primitive};
use derive_more::with_trait::Debug;
use std::ptr::NonNull;

/// A trait for types that record statistics about [`Move`]s.
pub trait Statistics {
    /// The stat type.
    type Stat: Stat;

    /// Returns the accumulated [`Self::Stat`]s about a [`Move`] in a [`Position`].
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value;

    /// Updates [`Self::Stat`]s for a [`Move`] in a [`Position`].
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value);
}

impl<T: Statistics> Statistics for &mut T {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        (*self).get(pos, m)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        (*self).update(pos, m, delta)
    }
}

impl<T: Statistics> Statistics for Option<T> {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.as_mut()
            .map_or_else(Default::default, |g| g.get(pos, m))
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        if let Some(g) = self {
            g.update(pos, m, delta);
        }
    }
}

impl<T: Statistics> Statistics for NonNull<T> {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.assume().get(pos, m)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.assume().update(pos, m, delta)
    }
}

/// A trait for statistics counters.
pub trait Stat {
    /// The value type.
    type Value: Primitive;

    /// Returns the current [`Self::Value`].
    fn get(&mut self) -> Self::Value;

    /// Updates and returns the current [`Self::Value`].
    fn update(&mut self, delta: Self::Value);
}

impl<T: Stat> Stat for &mut T {
    type Value = T::Value;

    #[inline(always)]
    fn get(&mut self) -> Self::Value {
        (*self).get()
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        (*self).update(delta);
    }
}

impl<T: Stat> Stat for Option<T> {
    type Value = T::Value;

    #[inline(always)]
    fn get(&mut self) -> Self::Value {
        self.as_mut().map_or_else(Default::default, Stat::get)
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        if let Some(s) = self {
            s.update(delta);
        }
    }
}

/// A saturating accumulator that implements the "gravity" formula.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Graviton(i8);

unsafe impl Integer for Graviton {
    type Repr = i8;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 127;
}

impl Stat for Graviton {
    type Value = i8;

    #[inline(always)]
    fn get(&mut self) -> Self::Value {
        self.0
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        let delta = delta.clamp(Self::MIN, Self::MAX) as i16;
        self.0 = (delta - delta.abs() * self.0 as i16 / Self::MAX as i16 + self.0 as i16) as i8
    }
}
