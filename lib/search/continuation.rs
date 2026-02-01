use crate::chess::{Move, PieceTo, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

#[derive(Debug, Zeroable)]
pub struct Reply(PieceTo<Graviton>);

impl const Default for Reply {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl const Statistics<Move> for Reply {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos.piece_on(wc).assume();
        self.0[piece as usize][wt as usize].get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos.piece_on(wc).assume();
        self.0[piece as usize][wt as usize].update(delta);
    }
}

#[derive(Debug, Zeroable)]
#[debug("Continuation")]
pub struct Continuation(PieceTo<[Reply; 2]>);

impl const Default for Continuation {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Continuation {
    #[inline(always)]
    pub const fn reply(&mut self, pos: &Position, m: Move) -> &mut Reply {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos.piece_on(wc).assume();
        let threats = pos.threats().contains(wt);
        &mut self.0[piece as usize][wt as usize][threats as usize]
    }
}
