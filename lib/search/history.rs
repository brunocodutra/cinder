use crate::chess::{Butterfly, Move, PieceTo, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Historical statistics about a [`Move`] in butterfly form.
#[derive(Debug, Zeroable)]
#[debug("ButterflyHistory")]
#[allow(clippy::type_complexity)]
pub struct ButterflyHistory([[[Butterfly<[[Graviton; 2]; 2]>; 2]; 2]; 2]);

impl Default for ButterflyHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl ButterflyHistory {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_ref(&self, pos: &Position, m: Move) -> &Graviton {
        let is_check = pos.is_check() as usize;
        let is_quiet = m.is_quiet() as usize;
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.attackers(wc).is_empty(), pos.attackers(wt).is_empty()];
        &self.0[pos.turn()][is_check][is_quiet][wc][wt][threats[0] as usize][threats[1] as usize]
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_mut(&mut self, pos: &Position, m: Move) -> &mut Graviton {
        let is_check = pos.is_check() as usize;
        let is_quiet = m.is_quiet() as usize;
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.attackers(wc).is_empty(), pos.attackers(wt).is_empty()];
        &mut self.0[pos.turn()][is_check][is_quiet][wc][wt][threats[0] as usize]
            [threats[1] as usize]
    }
}

impl Statistics<Move> for ButterflyHistory {
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

/// Historical statistics about [`Move`]s in relation to opposing king.
#[derive(Debug, Zeroable)]
#[debug("AttackerHistory")]
pub struct AttackerHistory([[PieceTo<[Graviton; 2]>; 64]; 2]);

impl Default for AttackerHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl AttackerHistory {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_ref(&self, pos: &Position, m: Move) -> &Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        let threat = pos.attackers(wt).is_empty();
        let ksq = pos.king(!pos.turn());
        &self.0[m.is_quiet() as usize][ksq][piece][wt][threat as usize]
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_mut(&mut self, pos: &Position, m: Move) -> &mut Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        let threat = pos.attackers(wt).is_empty();
        let ksq = pos.king(!pos.turn());
        &mut self.0[m.is_quiet() as usize][ksq][piece][wt][threat as usize]
    }
}

impl Statistics<Move> for AttackerHistory {
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

/// Historical statistics about [`Move`]s in relation to defending king.
#[derive(Debug, Zeroable)]
#[debug("DefenderHistory")]
pub struct DefenderHistory([[PieceTo<[Graviton; 2]>; 64]; 2]);

impl Default for DefenderHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl DefenderHistory {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_ref(&self, pos: &Position, m: Move) -> &Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        let threat = pos.attackers(wt).is_empty();
        let ksq = pos.king(pos.turn());
        &self.0[m.is_quiet() as usize][ksq][piece][wt][threat as usize]
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_mut(&mut self, pos: &Position, m: Move) -> &mut Graviton {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        let threat = pos.attackers(wt).is_empty();
        let ksq = pos.king(pos.turn());
        &mut self.0[m.is_quiet() as usize][ksq][piece][wt][threat as usize]
    }
}

impl Statistics<Move> for DefenderHistory {
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

impl Default for PieceToHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Statistics<Move> for PieceToHistory {
    type Stat = Graviton;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        self.0[piece][wt].get()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        self.0[piece][wt].update(delta);
    }
}

/// Historical statistics about [`Move`] continuations.
#[derive(Debug, Zeroable)]
#[debug("ContinuationHistory")]
pub struct ContinuationHistory([PieceTo<[PieceToHistory; 2]>; 2]);

impl Default for ContinuationHistory {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl ContinuationHistory {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn get(&mut self, pos: &Position, m: Move) -> &mut PieceToHistory {
        let (wc, wt) = (m.whence(), m.whither());
        let piece = pos[wc].piece().assume();
        let threat = pos.attackers(wt).is_empty();
        &mut self.0[m.is_quiet() as usize][piece][wt][threat as usize]
    }
}
