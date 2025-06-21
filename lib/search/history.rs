use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use derive_more::with_trait::Debug;

/// Historical statistics about a [`Move`].
#[derive(Debug)]
#[debug("History")]
pub struct History([[Butterfly<Graviton>; 2]; 2]);

impl Default for History {
    #[inline(always)]
    fn default() -> Self {
        Self([[[[Default::default(); 64]; 64]; 2]; 2])
    }
}

impl History {
    #[inline(always)]
    fn graviton(&mut self, pos: &Position, m: Move) -> &mut Graviton {
        let (wc, wt) = (m.whence() as usize, m.whither() as usize);
        &mut self.0[pos.turn() as usize][m.is_quiet() as usize][wc][wt]
    }
}

impl Statistics for History {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.graviton(pos, m).get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.graviton(pos, m).update(delta);
    }
}
