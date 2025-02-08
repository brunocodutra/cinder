use crate::chess::{Move, Position};
use crate::util::Assume;
use derive_more::with_trait::Debug;
use std::sync::atomic::{AtomicI8, Ordering::Relaxed};

#[cfg(test)]
use proptest::prelude::*;

pub trait Gravity {
    /// The history bonus.
    type Bonus;

    /// Returns the accumulated history [`Self::Bonus`] for a [`Move`].
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus;

    /// Update the [`Self::Bonus`] for a [`Move`].
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus);
}

/// The unit of [`Gravity`].
#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Graviton(#[cfg_attr(test, strategy(any::<i8>().prop_map_into()))] AtomicI8);

impl Gravity for Graviton {
    type Bonus = i8;

    #[inline(always)]
    fn get(&self, _: &Position, _: Move) -> Self::Bonus {
        self.0.load(Relaxed)
    }

    #[inline(always)]
    fn update(&self, _: &Position, _: Move, bonus: Self::Bonus) {
        let bonus = bonus.max(-i8::MAX);
        let result = self.0.fetch_update(Relaxed, Relaxed, |h| {
            Some((bonus as i16 - bonus.abs() as i16 * h as i16 / 127 + h as i16) as i8)
        });

        result.assume();
    }
}

impl<T: Gravity> Gravity for &T {
    type Bonus = T::Bonus;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        (*self).get(pos, m)
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        (*self).update(pos, m, bonus)
    }
}

impl<T: Gravity<Bonus: Default>> Gravity for Option<T> {
    type Bonus = T::Bonus;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        self.as_ref()
            .map_or_else(Default::default, |g| g.get(pos, m))
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        if let Some(g) = self {
            g.update(pos, m, bonus);
        }
    }
}
