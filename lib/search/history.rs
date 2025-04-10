use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use derive_more::with_trait::Debug;
use std::mem::MaybeUninit;

/// [Historical statistics] about a [`Move`].
///
/// [Historical statistics]: https://www.chessprogramming.org/History_Heuristic
#[derive(Debug)]
#[debug("History")]
pub struct History([[Butterfly<Graviton>; 2]; 2]);

impl Default for History {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl History {
    #[inline(always)]
    fn graviton(&self, pos: &Position, m: Move) -> &Graviton {
        let (wc, wt) = (m.whence() as usize, m.whither() as usize);
        &self.0[pos.turn() as usize][m.is_capture() as usize][wc][wt]
    }
}

impl Statistics for History {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.graviton(pos, m).get()
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.graviton(pos, m).update(delta);
    }
}
