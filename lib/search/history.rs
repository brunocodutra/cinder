use crate::chess::{Butterfly, Move, PieceTo, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Historical statistics about a [`Move`] in butterfly form.
#[derive(Debug, Zeroable)]
#[debug("ButterflyHistory")]
pub struct ButterflyHistory([[Butterfly<[[Graviton; 2]; 2]>; 2]; 2]);

impl const Default for ButterflyHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

const impl ButterflyHistory {
    #[inline(always)]
    fn graviton_ref(&self, pos: &Position, m: Move) -> &Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &self.0[pos.turn() as usize][m.is_quiet() as usize][wc as usize][wt as usize]
            [threats[0] as usize][threats[1] as usize]
    }

    #[inline(always)]
    fn graviton_mut(&mut self, pos: &Position, m: Move) -> &mut Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &mut self.0[pos.turn() as usize][m.is_quiet() as usize][wc as usize][wt as usize]
            [threats[0] as usize][threats[1] as usize]
    }
}

impl const Statistics<Move> for ButterflyHistory {
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

/// Historical statistics about a [`Move`] in piece-to form.
#[derive(Debug, Zeroable)]
#[debug("PieceToHistory")]
pub struct PieceToHistory(PieceTo<Graviton>);

impl const Default for PieceToHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl const Statistics<Move> for PieceToHistory {
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

/// Historical statistics about [`Move`]s in relation to opposing king.
#[derive(Debug, Zeroable)]
#[debug("AttackerHistory")]
pub struct AttackerHistory([[PieceToHistory; 64]; 2]);

impl const Default for AttackerHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl const Statistics<Move> for AttackerHistory {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        let ksq = pos.king(!pos.turn());
        self.0[m.is_quiet() as usize][ksq as usize].get(pos, m)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        let ksq = pos.king(!pos.turn());
        self.0[m.is_quiet() as usize][ksq as usize].update(pos, m, delta);
    }
}

/// Historical statistics about [`Move`]s in relation to defending king.
#[derive(Debug, Zeroable)]
#[debug("DefenderHistory")]
pub struct DefenderHistory([[PieceToHistory; 64]; 2]);

impl const Default for DefenderHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl const Statistics<Move> for DefenderHistory {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        let ksq = pos.king(pos.turn());
        self.0[m.is_quiet() as usize][ksq as usize].get(pos, m)
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        let ksq = pos.king(pos.turn());
        self.0[m.is_quiet() as usize][ksq as usize].update(pos, m, delta);
    }
}

/// Historical statistics about [`Move`] continuations.
#[derive(Debug, Zeroable)]
#[debug("ContinuationHistory")]
pub struct ContinuationHistory(PieceTo<[PieceToHistory; 2]>);

impl const Default for ContinuationHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

const impl ContinuationHistory {
    #[inline(always)]
    pub fn get(&mut self, pos: &Position, m: Move) -> &mut PieceToHistory {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos.piece_on(wc).assume();
        let threat = pos.threats().contains(wt);
        &mut self.0[piece as usize][wt as usize][threat as usize]
    }
}
