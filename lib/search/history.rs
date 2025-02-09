use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Gravity};
use derive_more::with_trait::Debug;
use std::mem::MaybeUninit;

/// [Historical statistics] about a [`Move`].
///
/// [Historical statistics]: https://www.chessprogramming.org/History_Heuristic
#[derive(Debug)]
#[debug("History")]
pub struct History([[[Butterfly<Graviton>; 2]; 2]; 2]);

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
        let turn = pos.turn() as usize;
        let is_check = pos.is_check() as usize;
        let is_capture = m.is_capture() as usize;
        &self.0[turn][is_check][is_capture][wc][wt]
    }
}

impl Gravity for History {
    type Bonus = <Graviton as Gravity>::Bonus;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        self.graviton(pos, m).get(pos, m)
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        self.graviton(pos, m).update(pos, m, bonus);
    }
}
