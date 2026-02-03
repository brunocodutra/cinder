use crate::chess::{Butterfly, Move, PieceTo, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
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

/// Historical statistics about a [`Move`] in relation to another.
#[derive(Debug, Zeroable)]
#[debug("ContinuationHistoryReply")]
pub struct ContinuationHistoryReply(PieceTo<Graviton>);

impl const Default for ContinuationHistoryReply {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl const Statistics<Move> for ContinuationHistoryReply {
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

/// Historical statistics about [`Move`] continuations.
#[derive(Debug, Zeroable)]
#[debug("ContinuationHistory")]
pub struct ContinuationHistory(PieceTo<[ContinuationHistoryReply; 2]>);

impl const Default for ContinuationHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl ContinuationHistory {
    #[inline(always)]
    pub const fn reply(&mut self, pos: &Position, m: Move) -> &mut ContinuationHistoryReply {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos.piece_on(wc).assume();
        let threats = pos.threats().contains(wt);
        &mut self.0[piece as usize][wt as usize][threats as usize]
    }
}
