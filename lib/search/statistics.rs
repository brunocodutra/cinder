use crate::chess::Position;
use crate::util::{Assume, Integer, Primitive};
use derive_more::with_trait::Debug;
use std::ptr::NonNull;

/// A trait for types that record statistics for [`Position`]s.
pub trait Statistics<C> {
    /// The stat type.
    type Stat: Stat;

    /// Returns the accumulated [`Self::Stat`]s.
    fn get(&mut self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value;

    /// Updates [`Self::Stat`]s.
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value);
}

impl<C, T: Statistics<C>> Statistics<C> for &mut T {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value {
        (*self).get(pos, ctx)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value) {
        (*self).update(pos, ctx, delta)
    }
}

impl<C, T: Statistics<C>> Statistics<C> for Option<T> {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value {
        self.as_mut()
            .map_or_else(Default::default, |g| g.get(pos, ctx))
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value) {
        if let Some(g) = self {
            g.update(pos, ctx, delta);
        }
    }
}

impl<C, T: Statistics<C>> Statistics<C> for NonNull<T> {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value {
        self.assume().get(pos, ctx)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value) {
        self.assume().update(pos, ctx, delta)
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

impl<T: Stat> Stat for NonNull<T> {
    type Value = T::Value;

    #[inline(always)]
    fn get(&mut self) -> Self::Value {
        self.assume().get()
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        self.assume().update(delta);
    }
}

/// A saturating accumulator that implements the "gravity" formula.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Graviton<const MIN: i16, const MAX: i16>(i16);

unsafe impl<const MIN: i16, const MAX: i16> Integer for Graviton<MIN, MAX> {
    type Repr = i16;
    const MIN: Self::Repr = MIN;
    const MAX: Self::Repr = MAX;
}

impl<const MIN: i16, const MAX: i16> Stat for Graviton<MIN, MAX> {
    type Value = <Self as Integer>::Repr;

    #[inline(always)]
    fn get(&mut self) -> Self::Value {
        const { assert!(MIN <= MAX) }
        self.0
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        let delta = delta.clamp(Self::MIN, Self::MAX) as i32;
        self.0 = (delta - delta.abs() * self.0 as i32 / Self::MAX as i32 + self.0 as i32) as i16
    }
}
