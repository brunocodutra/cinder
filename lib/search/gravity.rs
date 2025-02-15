use crate::util::Assume;
use derive_more::with_trait::Debug;
use std::sync::atomic::{AtomicI8, Ordering::Relaxed};

#[cfg(test)]
use proptest::prelude::*;

/// The unit of [`Gravity`].
#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Graviton(#[cfg_attr(test, strategy(any::<i8>().prop_map_into()))] AtomicI8);

impl Graviton {
    #[inline(always)]
    pub fn get(&self) -> i8 {
        self.0.load(Relaxed)
    }

    #[inline(always)]
    pub fn update(&self, bonus: i8) {
        let bonus = bonus.max(-i8::MAX);
        let result = self.0.fetch_update(Relaxed, Relaxed, |h| {
            Some((bonus as i16 - bonus.abs() as i16 * h as i16 / 127 + h as i16) as i8)
        });

        result.assume();
    }
}
