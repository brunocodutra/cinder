use crate::chess::{Butterfly, Move, Position};
use crate::search::{Counter, Stat, Statistics};
use derive_more::with_trait::Debug;
use std::mem::MaybeUninit;

/// Measures the effort spent searching a root [`Move`].
#[derive(Debug)]
#[debug("Attention")]
pub struct Attention(Butterfly<Counter>);

impl Default for Attention {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl Attention {
    #[inline(always)]
    pub fn nodes(&self, m: Move) -> &Counter {
        &self.0[m.whence() as usize][m.whither() as usize]
    }
}

impl Statistics for Attention {
    type Stat = Counter;

    #[inline(always)]
    fn get(&self, _: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.nodes(m).get()
    }

    #[inline(always)]
    fn update(&self, _: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.nodes(m).update(delta);
    }
}
