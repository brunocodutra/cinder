use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Historical statistics about a [`Move`].
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[debug("History")]
pub struct History(
    #[allow(clippy::type_complexity)]
    [[Butterfly<[[<History as Statistics<Move>>::Stat; 2]; 2]>; 2]; 2],
);

impl Default for History {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl History {
    pub const LIMIT: i16 = 256;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton(&mut self, pos: &Position, m: Move) -> &mut <Self as Statistics<Move>>::Stat {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &mut self.0[pos.turn() as usize][m.is_quiet() as usize][wc as usize][wt as usize]
            [threats[0] as usize][threats[1] as usize]
    }
}

impl Statistics<Move> for History {
    type Stat = Graviton<{ -Self::LIMIT }, { Self::LIMIT }>;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.graviton(pos, m).get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.graviton(pos, m).update(delta);
    }
}
