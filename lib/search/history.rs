use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Rating};
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

impl Rating for History {
    type Bonus = i8;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        self.graviton(pos, m).get()
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        self.graviton(pos, m).update(bonus);
    }
}
