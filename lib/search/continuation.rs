use crate::chess::{Move, PieceTo, Position};
use crate::search::{History, Stat, Statistics};
use crate::util::Assume;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
pub struct Reply([[<Reply as Statistics<Move>>::Stat; 64]; 6]);

impl Default for Reply {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Reply {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton(&mut self, pos: &Position, m: Move) -> &mut <Self as Statistics<Move>>::Stat {
        let (wc, wt) = (m.whence(), m.whither());
        let role = pos.role_on(wc).assume() as usize;
        &mut self.0[role][wt as usize]
    }
}

impl Statistics<Move> for Reply {
    type Stat = <History as Statistics<Move>>::Stat;

    #[inline(always)]
    fn get(&mut self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.graviton(pos, m).get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.graviton(pos, m).update(delta);
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[debug("Continuation")]
pub struct Continuation(PieceTo<[Reply; 2]>);

impl Default for Continuation {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Continuation {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn reply(&mut self, pos: &Position, m: Move) -> &mut Reply {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos.piece_on(wc).assume();
        let threats = pos.threats().contains(wt);
        &mut self.0[piece as usize][wt as usize][threats as usize]
    }
}
