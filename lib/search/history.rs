use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Historical statistics about a [`Move`].
#[derive(Debug, Zeroable)]
#[debug("History")]
pub struct History([[Butterfly<[[Graviton; 2]; 2]>; 2]; 2]);

impl const Default for History {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl History {
    #[inline(always)]
    const fn graviton_ref(&self, pos: &Position, m: Move) -> &Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &self.0[pos.turn() as usize][m.is_quiet() as usize][wc as usize][wt as usize]
            [threats[0] as usize][threats[1] as usize]
    }

    #[inline(always)]
    const fn graviton_mut(&mut self, pos: &Position, m: Move) -> &mut Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &mut self.0[pos.turn() as usize][m.is_quiet() as usize][wc as usize][wt as usize]
            [threats[0] as usize][threats[1] as usize]
    }
}

impl const Statistics<Move> for History {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.graviton_ref(pos, m).get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.graviton_mut(pos, m).update(delta);
    }
}
