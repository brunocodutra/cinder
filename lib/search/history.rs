use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Gravity};
use derive_more::with_trait::Debug;

/// [Historical statistics] about a [`Move`].
///
/// [Historical statistics]: https://www.chessprogramming.org/History_Heuristic
#[derive(Debug)]
#[debug("History")]
pub struct History(Box<[[Butterfly<Graviton>; 2]; 2]>);

impl Default for History {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { Box::new_zeroed().assume_init() })
    }
}

impl Gravity for History {
    type Bonus = <Graviton as Gravity>::Bonus;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        let (wc, wt) = (m.whence() as usize, m.whither() as usize);
        self.0[pos.turn() as usize][m.is_capture() as usize][wc][wt].get(pos, m)
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        let (wc, wt) = (m.whence() as usize, m.whither() as usize);
        self.0[pos.turn() as usize][m.is_capture() as usize][wc][wt].update(pos, m, bonus);
    }
}
