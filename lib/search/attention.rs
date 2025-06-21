use crate::chess::{Butterfly, Move};
use derive_more::with_trait::Debug;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
use proptest::prelude::*;

/// A linear node counter.
#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Nodes(#[cfg_attr(test, strategy(any::<usize>().prop_map_into()))] AtomicUsize);

impl Nodes {
    #[inline(always)]
    pub fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn increment(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// Measures the effort spent searching a root [`Move`].
#[derive(Debug)]
#[debug("Attention")]
pub struct Attention(Butterfly<Nodes>);

impl Default for Attention {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl Attention {
    #[inline(always)]
    pub fn nodes(&self, m: Move) -> &Nodes {
        &self.0[m.whence() as usize][m.whither() as usize]
    }
}
