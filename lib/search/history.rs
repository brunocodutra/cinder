use crate::chess::{Butterfly, Color, Move, Piece, PieceTo, Position, Square};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Context about a [`Position`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PositionContext {
    pub turn: Color,
    pub is_check: bool,
    pub our_king: Square,
    pub their_king: Square,
}

impl PositionContext {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(pos: &Position) -> Self {
        let turn = pos.turn();
        let our_king = pos.king(turn);

        PositionContext {
            turn,
            is_check: !pos.attackers(our_king).is_empty(),
            our_king,
            their_king: pos.king(!turn),
        }
    }
}

/// Context about a [`Move`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(clippy::struct_excessive_bools)]
pub struct MoveContext {
    pub turn: Color,
    pub piece: Piece,
    pub whence: Square,
    pub whither: Square,
    pub is_quiet: bool,
    pub is_check: bool,
    pub our_king: Square,
    pub their_king: Square,
    pub whence_attacked: bool,
    pub whither_attacked: bool,
}

impl MoveContext {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(ctx: &PositionContext, pos: &Position, m: Move) -> Self {
        let (wc, wt) = (m.whence(), m.whither());

        MoveContext {
            turn: ctx.turn,
            piece: pos[wc].piece().assume(),
            whence: wc,
            whither: wt,
            is_quiet: m.is_quiet(),
            is_check: ctx.is_check,
            our_king: ctx.our_king,
            their_king: ctx.their_king,
            whence_attacked: !pos.attackers(wc).is_empty(),
            whither_attacked: !pos.attackers(wt).is_empty(),
        }
    }
}

/// Historical statistics about a [`Move`] in butterfly form.
#[derive(Debug, Zeroable)]
#[debug("ButterflyHistory")]
#[expect(clippy::type_complexity)]
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
    fn graviton_ref(&self, ctx: &MoveContext) -> &Graviton {
        &self.0[ctx.turn][ctx.is_check as usize][ctx.is_quiet as usize][ctx.whence][ctx.whither]
            [ctx.whence_attacked as usize][ctx.whither_attacked as usize]
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_mut(&mut self, ctx: &MoveContext) -> &mut Graviton {
        &mut self.0[ctx.turn][ctx.is_check as usize][ctx.is_quiet as usize][ctx.whence][ctx.whither]
            [ctx.whence_attacked as usize][ctx.whither_attacked as usize]
    }
}

impl Statistics<MoveContext> for ButterflyHistory {
    type Stat = Graviton;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn get(&self, ctx: MoveContext) -> <Self::Stat as Stat>::Value {
        self.graviton_ref(&ctx).get()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, ctx: MoveContext, delta: <Self::Stat as Stat>::Value) {
        self.graviton_mut(&ctx).update(delta);
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
    fn graviton_ref(&self, ctx: &MoveContext) -> &Graviton {
        &self.0[ctx.is_quiet as usize][ctx.their_king][ctx.piece][ctx.whither]
            [ctx.whither_attacked as usize]
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_mut(&mut self, ctx: &MoveContext) -> &mut Graviton {
        &mut self.0[ctx.is_quiet as usize][ctx.their_king][ctx.piece][ctx.whither]
            [ctx.whither_attacked as usize]
    }
}

impl Statistics<MoveContext> for AttackerHistory {
    type Stat = Graviton;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn get(&self, ctx: MoveContext) -> <Self::Stat as Stat>::Value {
        self.graviton_ref(&ctx).get()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, ctx: MoveContext, delta: <Self::Stat as Stat>::Value) {
        self.graviton_mut(&ctx).update(delta);
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
    fn graviton_ref(&self, ctx: &MoveContext) -> &Graviton {
        &self.0[ctx.is_quiet as usize][ctx.our_king][ctx.piece][ctx.whither]
            [ctx.whither_attacked as usize]
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn graviton_mut(&mut self, ctx: &MoveContext) -> &mut Graviton {
        &mut self.0[ctx.is_quiet as usize][ctx.our_king][ctx.piece][ctx.whither]
            [ctx.whither_attacked as usize]
    }
}

impl Statistics<MoveContext> for DefenderHistory {
    type Stat = Graviton;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn get(&self, ctx: MoveContext) -> <Self::Stat as Stat>::Value {
        self.graviton_ref(&ctx).get()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, ctx: MoveContext, delta: <Self::Stat as Stat>::Value) {
        self.graviton_mut(&ctx).update(delta);
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

impl Statistics<MoveContext> for PieceToHistory {
    type Stat = Graviton;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn get(&self, ctx: MoveContext) -> <Self::Stat as Stat>::Value {
        self.0[ctx.piece][ctx.whither].get()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, ctx: MoveContext, delta: <Self::Stat as Stat>::Value) {
        self.0[ctx.piece][ctx.whither].update(delta);
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
        let attacked = !pos.attackers(wt).is_empty();
        &mut self.0[m.is_quiet() as usize][piece][wt][attacked as usize]
    }
}
