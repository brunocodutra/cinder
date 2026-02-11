use crate::chess::Position;
use crate::util::{Assume, Float, Num, zero};
use bytemuck::{NoUninit, Zeroable};
use derive_more::with_trait::Debug;
use std::{marker::Destruct, ptr::NonNull};

/// A trait for types that record statistics for [`Position`]s.
pub const trait Statistics<C> {
    /// The stat type.
    type Stat: Stat;

    /// Returns the accumulated [`Self::Stat`]s.
    fn get(&self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value;

    /// Updates [`Self::Stat`]s.
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value);
}

impl<C, T: [const] Statistics<C>> const Statistics<C> for &mut T {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value {
        (**self).get(pos, ctx)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value) {
        (**self).update(pos, ctx, delta);
    }
}

impl<C, T> const Statistics<C> for Option<T>
where
    C: [const] Destruct,
    T: [const] Statistics<C, Stat: Stat<Value: [const] Destruct>>,
{
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value {
        match self {
            Some(g) => g.get(pos, ctx),
            None => zero(),
        }
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value) {
        if let Some(g) = self {
            g.update(pos, ctx, delta);
        }
    }
}

impl<C, T: [const] Statistics<C>> const Statistics<C> for NonNull<T> {
    type Stat = T::Stat;

    #[inline(always)]
    fn get(&self, pos: &Position, ctx: C) -> <Self::Stat as Stat>::Value {
        self.assume().get(pos, ctx)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, ctx: C, delta: <Self::Stat as Stat>::Value) {
        self.assume().update(pos, ctx, delta);
    }
}

/// A trait for statistics counters.
pub const trait Stat {
    /// The value type.
    type Value: Zeroable;

    /// Returns the current [`Self::Value`].
    fn get(&self) -> Self::Value;

    /// Updates and returns the current [`Self::Value`].
    fn update(&mut self, delta: Self::Value);
}

impl<T: [const] Stat<Value: [const] Destruct>> const Stat for &mut T {
    type Value = T::Value;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        (**self).get()
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        (**self).update(delta);
    }
}

impl<T: [const] Stat<Value: [const] Destruct>> const Stat for Option<T> {
    type Value = T::Value;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.as_ref().map_or_else(zero, Stat::get)
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        if let Some(s) = self {
            s.update(delta);
        }
    }
}

impl<T: [const] Stat<Value: [const] Destruct>> const Stat for NonNull<T> {
    type Value = T::Value;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.assume().get()
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        self.assume().update(delta);
    }
}

/// A saturating accumulator that implements the "gravity" formula.
#[derive(Debug, Copy, Zeroable, NoUninit)]
#[derive_const(Default, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Graviton(f32);

unsafe impl const Num for Graviton {
    type Repr = f32;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 1.0;
}

unsafe impl const Float for Graviton {}

impl const Stat for Graviton {
    type Value = <Self as Num>::Repr;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        self.0
    }

    #[inline(always)]
    fn update(&mut self, delta: Self::Value) {
        self.0 += delta.abs().mul_add(-self.0, delta);
        self.0 = self.0.clip(Self::MIN, Self::MAX);
    }
}
