use crate::util::{Assume, Int, Num};
use crate::{chess::*, simd::*};
use bytemuck::zeroed;
use derive_more::with_trait::{Debug, Deref, Display, Error, From, IntoIterator};
use std::fmt::{self, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, Shl, Sub};
use std::{num::NonZeroU32, str::FromStr};

#[cfg(test)]
use proptest::{prelude::*, sample::Selector};

#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Error, From)]
struct CapacityError;

#[derive(Debug, Default)]
struct NoCapacity;

impl MoveCollector for NoCapacity {
    type Error = CapacityError;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_one(&mut self, _: Move) -> Result<(), CapacityError> {
        Err(CapacityError)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_attacks(
        &mut self,
        pos: &Position,
        indices: IdxSet,
        targets: Bitboard,
    ) -> Result<(), Self::Error> {
        if targets & pos.pins().attacks().matching(indices) != zeroed() {
            Err(CapacityError)
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_pawn_promotions(
        &mut self,
        pos: &Position,
        targets: Bitboard,
    ) -> Result<(), Self::Error> {
        self.collect_pawn_pushes(pos, targets)?;

        let turn = pos.turn();
        let pawns = pos.roles()[turn].matching(Some(Role::Pawn));
        self.collect_attacks(pos, pawns.into(), targets & pos.by_color(!turn))
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_pawn_pushes(
        &mut self,
        pos: &Position,
        targets: Bitboard,
    ) -> Result<(), Self::Error> {
        let turn = pos.turn();
        let unpinned_pushes = pos.king(turn).file().bitboard() | pos.pins().unpinned();
        let pawns = unpinned_pushes & pos.by_piece(Piece::new(Role::Pawn, turn));
        let vacant = Bitboard::from(pos.vacant()).perspective(turn);

        let third = Rank::Third.bitboard();
        let single = pawns.perspective(turn).shl(8) & vacant;
        let double = single.bitand(third).shl(8) & vacant;
        if targets.perspective(turn) & (single | double) != zeroed() {
            Err(CapacityError)
        } else {
            Ok(())
        }
    }
}

trait MoveGen<C: MoveCollector> {
    fn moves(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error>;
    fn noisy(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error>;
}

enum MovesGenerator<const CHECKS: usize> {}

impl<const CHECKS: usize> MovesGenerator<CHECKS> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn en_passant<C: MoveCollector>(
        pos: &Position,
        wt: Bitboard,
        collector: &mut C,
    ) -> Result<(), C::Error> {
        const { assert!(CHECKS < 2) }

        let turn = pos.turn();
        let ksq = pos.king(turn);
        let pawns = pos.roles()[turn].matching(Some(Role::Pawn));
        if let Some(wt) = pos.en_passant().filter(|ep| wt.contains(*ep)) {
            for idx in pos.pins().attacks()[wt] & pawns {
                let wc = pos.squares()[turn][idx].assume();
                let mut placement = *pos.placement();
                placement.set(wt, pos[wc]);
                placement.set(wc, zeroed());
                placement.set(Square::new(wt.file(), wc.rank()), zeroed());

                let rays = ksq.rays();
                let furled = placement.furl(rays);
                let theirs = furled.by_color(!turn);
                if !(theirs & furled.visible() & furled.attackers() & rays.valid()).any() {
                    collector.collect_one(Move::capture(wc, wt, None))?;
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn castling<C: MoveCollector>(
        pos: &Position,
        wt: Bitboard,
        collector: &mut C,
    ) -> Result<(), C::Error> {
        const { assert!(CHECKS == 0) }

        let turn = pos.turn();
        let occ = pos.occupied();
        let ksq = pos.king(turn);
        for castling in [Square::C1.perspective(turn), Square::G1.perspective(turn)] {
            if wt.contains(castling) && pos.castles().has(castling) {
                let rook = Castles::rook(castling).assume().whence();
                if Bitboard::segment(ksq, rook) & occ == zeroed() {
                    let threats = pos.threats()[!turn].to_simd();
                    let path = Bitboard::segment(ksq, castling).with(castling);
                    if path & threats.simd_ne(zeroed()) == zeroed() {
                        collector.collect_one(Move::regular(ksq, castling, None))?;
                    }
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn evasions<C: MoveCollector>(
        pos: &Position,
        mut wt: Bitboard,
        collector: &mut C,
    ) -> Result<(), C::Error> {
        let turn = pos.turn();
        let wc = pos.king(turn);

        for idx in pos.checkers() {
            let checker = pos.roles()[!turn][idx].assume();
            if matches!(checker, Role::Bishop | Role::Rook | Role::Queen) {
                let sq = pos.squares()[!turn][idx].assume();
                wt &= !Bitboard::line(wc, sq).without(sq);
            }
        }

        let threats = pos.threats()[!turn].to_simd().simd_ne(zeroed());
        collector.collect_attacks(pos, Idx::KING.to_set(), wt & !threats & !pos.by_color(turn))
    }
}

impl<C: MoveCollector> MoveGen<C> for MovesGenerator<0> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn moves(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error> {
        let turn = pos.turn();
        let ours = pos.by_color(turn);
        let theirs = pos.by_color(!turn);
        let eighth = Rank::Eighth.perspective(turn).bitboard();
        let pawns = pos.roles()[turn].matching(Some(Role::Pawn));
        collector.collect_pawn_pushes(pos, wt & !eighth)?;
        collector.collect_pawn_promotions(pos, wt & eighth)?;
        collector.collect_attacks(pos, pawns.into(), wt & theirs & !eighth)?;

        let none = pos.roles()[turn].matching(None);
        let not_king_nor_pawns = !Idx::KING.to_set() & !(pawns | none);
        collector.collect_attacks(pos, not_king_nor_pawns, wt & !ours)?;

        Self::en_passant(pos, wt, collector)?;
        Self::evasions(pos, wt, collector)?;
        Self::castling(pos, wt, collector)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn noisy(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error> {
        let turn = pos.turn();
        let theirs = pos.by_color(!turn);
        let eighth = Rank::Eighth.perspective(turn).bitboard();
        let pawns = pos.roles()[turn].matching(Some(Role::Pawn));
        collector.collect_pawn_promotions(pos, wt & eighth)?;
        collector.collect_attacks(pos, pawns.into(), wt & theirs & !eighth)?;

        let none = pos.roles()[turn].matching(None);
        let not_king_nor_pawns = !Idx::KING.to_set() & !(pawns | none);
        collector.collect_attacks(pos, not_king_nor_pawns, wt & theirs)?;

        Self::en_passant(pos, wt, collector)?;
        Self::evasions(pos, wt & theirs, collector)
    }
}

impl<C: MoveCollector> MoveGen<C> for MovesGenerator<1> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn moves(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error> {
        let turn = pos.turn();
        let ksq = pos.king(turn);
        let checks = pos.checkers().iter().fold(Bitboard::empty(), |bb, idx| {
            let sq = pos.squares()[!turn][idx].assume();
            Bitboard::segment(ksq, sq).with(sq) | bb
        });

        let ours = pos.by_color(turn);
        let theirs = pos.by_color(!turn);
        let pawns = pos.roles()[turn].matching(Some(Role::Pawn));
        let eighth = Rank::Eighth.perspective(turn).bitboard();
        collector.collect_pawn_pushes(pos, wt & checks & !eighth)?;
        collector.collect_pawn_promotions(pos, wt & checks & eighth)?;
        collector.collect_attacks(pos, pawns.into(), wt & checks & theirs & !eighth)?;

        let none = pos.roles()[turn].matching(None);
        let not_king_nor_pawns = !Idx::KING.to_set() & !(pawns | none);
        collector.collect_attacks(pos, not_king_nor_pawns, wt & checks & !ours)?;

        if let Some(wt) = pos.en_passant().filter(|ep| wt.contains(*ep)) {
            if checks.contains(wt.perspective(turn).sub(8).perspective(turn)) {
                Self::en_passant(pos, wt.bitboard(), collector)?;
            }
        }

        Self::evasions(pos, wt, collector)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn noisy(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error> {
        let turn = pos.turn();
        let ksq = pos.king(turn);
        let theirs = pos.by_color(!turn);
        let checks = pos.checkers().iter().fold(Bitboard::empty(), |bb, idx| {
            let sq = pos.squares()[!turn][idx].assume();
            Bitboard::segment(ksq, sq).with(sq) | bb
        });

        let eighth = Rank::Eighth.perspective(turn).bitboard();
        collector.collect_pawn_promotions(pos, wt & !theirs & eighth & checks)?;
        let ep = pos.en_passant().map_or_else(zeroed, Square::bitboard);
        Self::moves(pos, wt & (ep | theirs), collector)
    }
}

impl<C: MoveCollector> MoveGen<C> for MovesGenerator<2> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn moves(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error> {
        Self::evasions(pos, wt, collector)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn noisy(pos: &Position, wt: Bitboard, collector: &mut C) -> Result<(), C::Error> {
        Self::evasions(pos, wt & pos.by_color(!pos.turn()), collector)
    }
}

/// The current position on the board.
///
/// This type guarantees that it only holds valid positions.
#[derive(Debug, Clone, Copy, Eq, Deref)]
#[debug("Position({self})")]
pub struct Position {
    #[deref(forward)]
    board: Board,
    pins: Pins,
    threats: Threats,
    zobrists: Zobrists,
    direct_checks: [Bitboard; 4],
    history: [Aligned<[u32; 32]>; 2],
}

#[cfg(test)]
impl Arbitrary for Position {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        (0..128, any::<Selector>())
            .prop_map(|(moves, selector)| {
                let mut pos = Position::default();

                for _ in 0..moves {
                    if pos.outcome().is_none() {
                        pos.play(selector.select(pos.moves()));
                    } else {
                        break;
                    }
                }

                pos
            })
            .no_shrink()
            .boxed()
    }
}

impl Default for Position {
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn default() -> Self {
        let board = Board::default();
        let threats = board.threats();

        Self {
            pins: board.pins(&threats),
            threats,
            zobrists: board.zobrists(),
            direct_checks: zeroed(),
            history: zeroed(),
            board,
        }
    }
}

impl Hash for Position {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.zobrists.hash.hash(state);
    }
}

impl PartialEq for Position {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.zobrists.hash.eq(&other.zobrists.hash) && self.board.eq(&other.board)
    }
}

impl Position {
    /// The side to move.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn turn(&self) -> Color {
        self.board.turn
    }

    /// The number of halfmoves since the last capture or pawn advance.
    ///
    /// It resets to 0 whenever a piece is captured or a pawn is moved.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn halfmoves(&self) -> u8 {
        self.board.halfmoves
    }

    /// The current move number since the start of the game.
    ///
    /// It starts at 1, and is incremented after every move by black.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn fullmoves(&self) -> NonZeroU32 {
        self.board.fullmoves.convert().assume()
    }

    /// The en passant square.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn en_passant(&self) -> Option<Square> {
        self.board.en_passant
    }

    /// The castle rights.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn castles(&self) -> Castles {
        self.board.castles
    }

    /// This position's [zobrist hashes](`Zobrists`).
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn zobrists(&self) -> &Zobrists {
        &self.zobrists
    }

    /// The board placement.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn placement(&self) -> &Placement {
        self.board.placement()
    }

    /// Squares by piece [`Idx`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn squares(&self) -> &SquareByIdx {
        self.board.squares()
    }

    /// Roles by piece [`Idx`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn roles(&self) -> &RoleByIdx {
        self.board.roles()
    }

    /// The [`Pins`] in this position.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pins(&self) -> &Pins {
        &self.pins
    }

    /// The [`Threats`] in this position.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn threats(&self) -> &Threats {
        &self.threats
    }

    /// [`IdxSet`] of pieces defending a square.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn defenders(&self, sq: Square) -> IdxSet {
        let turn = self.turn();
        self.threats[turn as usize][sq]
    }

    /// [`IdxSet`] of pieces attacking a square.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attackers(&self, sq: Square) -> IdxSet {
        let turn = self.turn();
        self.threats[!turn as usize][sq]
    }

    /// [`IdxSet`] of pieces giving check.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn checkers(&self) -> IdxSet {
        self.attackers(self.king(self.turn()))
    }

    /// [`Square`] occupied by a the king of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn king(&self, side: Color) -> Square {
        self.board.king(side).assume()
    }

    /// Game [`Phase`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn phase(&self) -> Phase {
        self.board.phase()
    }

    /// [`Square`]s occupied.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn occupied(&self) -> M8x64 {
        self.board.occupied()
    }

    /// [`Square`]s vacant.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn vacant(&self) -> M8x64 {
        self.board.vacant()
    }

    /// [`Square`]s occupied by a [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_piece(&self, piece: Piece) -> M8x64 {
        self.board.by_piece(piece)
    }

    /// [`Square`]s occupied by pieces of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_color(&self, side: Color) -> M8x64 {
        self.board.by_color(side)
    }

    /// [`Square`]s occupied by pieces of a [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_role(&self, role: Role) -> M8x64 {
        self.board.by_role(role)
    }

    /// [`Square`]s occupied by pawns of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pawns(&self, side: Color) -> M8x64 {
        self.by_piece(Piece::new(Role::Pawn, side))
    }

    /// Whether the game is a draw by the 50-move rule.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_draw_by_50_move_rule(&self) -> bool {
        self.halfmoves() >= 100
    }

    /// Whether this position has insufficient material.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_material_insufficient(&self) -> bool {
        use {Piece::*, Role::*};
        match self.occupied().count() {
            2 => true,
            3 => (self.by_role(Bishop) | self.by_role(Knight)).any(),
            4 => {
                let wb = self.by_piece(WhiteBishop);
                let bb = self.by_piece(BlackBishop);

                let dark = Bitboard::dark();
                let light = Bitboard::light();

                !(light.bitand(wb).is_empty() || light.bitand(bb).is_empty())
                    || !(dark.bitand(wb).is_empty() || dark.bitand(bb).is_empty())
            }
            _ => false,
        }
    }

    /// Whether this position is a check.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_check(&self) -> bool {
        !self.checkers().is_empty()
    }

    /// Whether this position is a checkmate.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_checkmate(&self) -> bool {
        match self.checkers().len() {
            0 => false,
            1 => MovesGenerator::<1>::moves(self, Bitboard::full(), &mut NoCapacity).is_ok(),
            _ => MovesGenerator::<2>::moves(self, Bitboard::full(), &mut NoCapacity).is_ok(),
        }
    }

    /// Whether this position is a stalemate.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_stalemate(&self) -> bool {
        !self.is_check()
            && MovesGenerator::<0>::moves(self, Bitboard::full(), &mut NoCapacity).is_ok()
    }

    /// Whether the game is a draw by repetition.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_draw_by_repetition(&self) -> bool {
        let hash @ 1.. = self.zobrists().hash.cast() else {
            return false;
        };

        let history: u32x32 = self.history[self.turn()].cast();
        history.simd_eq(Simd::splat(hash)).any()
    }

    /// The [`Outcome`] of the game in case this position is final.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn outcome(&self) -> Option<Outcome> {
        if self.is_checkmate() {
            Some(Outcome::Checkmate(!self.turn()))
        } else if self.is_stalemate() {
            Some(Outcome::Stalemate)
        } else if self.is_draw_by_50_move_rule() {
            Some(Outcome::DrawBy50MoveRule)
        } else if self.is_draw_by_repetition() {
            Some(Outcome::DrawByThreefoldRepetition)
        } else if self.is_material_insufficient() {
            Some(Outcome::DrawByInsufficientMaterial)
        } else {
            None
        }
    }

    /// Whether a [`Move`] checks the opposing king directly.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn gives_direct_check(&self, m: Move) -> bool {
        let (wc, wt) = (m.whence(), m.whither());
        let role = match m.promotion() {
            None => self[wc].role().assume(),
            Some(r) => r,
        };

        use Role::*;
        if role == Queen {
            let checking = self.direct_checks[Bishop as usize] | self.direct_checks[Rook as usize];
            checking.contains(wt)
        } else if role != Role::King {
            self.direct_checks.get(role as usize).assume().contains(wt)
        } else if (wt - wc).abs() == 2 {
            let wt = Castles::rook(wt).assume().whither();
            self.direct_checks[Rook as usize].contains(wt)
        } else {
            false
        }
    }

    /// The legal moves to a set of destinations that can be played in this position.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn moves_to(&self, wt: Bitboard) -> Moves {
        let mut moves = Moves::default();
        match self.checkers().len() {
            0 => MovesGenerator::<0>::moves(self, wt, &mut moves).assume(),
            1 => MovesGenerator::<1>::moves(self, wt, &mut moves).assume(),
            _ => MovesGenerator::<2>::moves(self, wt, &mut moves).assume(),
        }

        moves
    }

    /// The legal moves that can be played in this position.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn moves(&self) -> Moves {
        self.moves_to(Bitboard::full())
    }

    /// The legal noisy moves to a set of destinations that can be played in this position.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn noisy_to(&self, wt: Bitboard) -> Moves {
        let mut moves = Moves::default();
        match self.checkers().len() {
            0 => MovesGenerator::<0>::noisy(self, wt, &mut moves).assume(),
            1 => MovesGenerator::<1>::noisy(self, wt, &mut moves).assume(),
            _ => MovesGenerator::<2>::noisy(self, wt, &mut moves).assume(),
        }

        moves
    }

    /// The legal noisy moves that can be played in this position.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn noisy(&self) -> Moves {
        self.noisy_to(Bitboard::full())
    }

    /// The sequence of captures on a square starting from a move ordered by least valued captor.
    #[inline(always)]
    pub fn exchanges(&self, m: Move) -> impl Iterator<Item = (Move, Role)> {
        use {Rank::*, Role::*};

        #[inline(always)]
        gen move {
            let (wc, wt) = (m.whence(), m.whither());
            if (self.attackers(wt).is_empty() && self.attackers(wc).is_empty())
                || self[wc].role() == Some(King)
            {
                return;
            }

            let mut turn = self.turn();
            let kings = [self.king(Color::White), self.king(Color::Black)];

            let mut placement = *self.placement();
            placement.set(wt, self[wc]);
            placement.set(wc, zeroed());
            if placement[wt].is_empty() && m.is_capture() {
                placement.set(Square::new(wt.file(), wc.rank()), zeroed());
            }

            loop {
                turn = !turn;

                let rays = wt.rays();
                let furled = placement.furl(rays);
                let attackers = furled.visible() & furled.attackers() & rays.valid();
                let candidates = furled.by_color(turn) & attackers;
                if !candidates.any() {
                    return;
                }

                let unpinned = {
                    let rays = kings[turn].rays();
                    let furled = placement.furl(rays);
                    let theirs = furled.by_color(!turn);
                    let visible = furled.visible();
                    let attackers = theirs & visible & furled.attackers() & rays.valid();
                    let line = Bitboard::line(kings[turn], wt).with(wt);
                    if cfg!(target_feature = "avx512f") || attackers.any() {
                        if !line & attackers.unfurl(rays) != zeroed() {
                            return;
                        }
                    }

                    let pins = rays.pins();
                    let ours = furled.by_color(turn);
                    let nearest = ours & visible & pins;
                    let beyond = furled.blend(nearest, zeroed()).visible() & pins;
                    let pinners = beyond & furled.pinners() & theirs;
                    let pinned = nearest & pinners.flood_ranks();

                    if cfg!(target_feature = "avx512f") || pinned.any() {
                        line | !pinned.unfurl(rays)
                    } else {
                        Bitboard::full()
                    }
                };

                let candidates = unpinned & (candidates.unfurl(rays) & rays.inv().valid());
                if candidates.is_empty() {
                    return;
                }

                let roles = u64x8::from_array([
                    placement.by_role(Pawn).to_bitmask(),
                    placement.by_role(Knight).to_bitmask(),
                    placement.by_role(Bishop).to_bitmask(),
                    placement.by_role(Rook).to_bitmask(),
                    placement.by_role(Queen).to_bitmask(),
                    kings[turn].bitboard().get(),
                    0,
                    0,
                ]);

                let roles = roles & Simd::splat(*candidates);
                let captor = Role::new(roles.simd_ne(zeroed()).to_bitmask().trailing_zeros() as u8);
                let captors = roles.as_array().get(captor as usize).assume();
                let wc = <Square as Num>::new(captors.trailing_zeros() as i8);

                if captor == King {
                    let theirs = furled.by_color(!turn);
                    if theirs.bitand(attackers).any() {
                        return;
                    }
                }

                let promotion = if captor == Pawn && wt.rank().perspective(turn) == Eighth {
                    Some(Queen)
                } else {
                    None
                };

                placement.set(wt, placement[wc]);
                placement.set(wc, zeroed());
                yield (Move::capture(wc, wt, promotion), captor);
            }
        }
    }

    /// Whether a [`Move`] is legal in this position.
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn is_legal(&self, m: Move) -> bool {
        use {Rank::*, Role::*};

        let turn = self.turn();
        let occ = self.occupied();
        let ours = self.by_color(turn);
        let ksq = self.king(turn);

        let (wc, wt) = (m.whence(), m.whither());
        let unpinned = match self.checkers().len() {
            0 => Bitboard::from(ours) & (Bitboard::line(ksq, wt) | self.pins().unpinned()),
            1 => Bitboard::from(ours) & self.pins().unpinned(),
            2 => ksq.bitboard(),
            _ => return false,
        };

        if !unpinned.contains(wc) || ours.test(wt.cast()) || self[wt].role() == Some(King) {
            return false;
        }

        let piece = self[wc].piece().assume();
        if m.is_promotion() != ((piece.role(), wt.rank()) == (Pawn, Eighth.perspective(turn))) {
            return false;
        }

        if self.en_passant() == Some(wt) && m.is_capture() && piece.role() == Pawn {
            let mut placement = *self.placement();
            placement.set(wt, self[wc]);
            placement.set(wc, zeroed());
            placement.set(Square::new(wt.file(), wc.rank()), zeroed());

            let rays = ksq.rays();
            let furled = placement.furl(rays);
            let theirs = furled.by_color(!turn);
            let attackers = theirs & furled.visible() & furled.attackers() & rays.valid();
            return !attackers.any() && self.threats()[turn][wt].contains(self[wc].idx().assume());
        }

        if m.is_capture() == self[wt].is_empty() {
            return false;
        }

        if piece.role() == King && (wt - wc).abs() == 2 {
            let path = Bitboard::segment(ksq, wt).with(wt);
            let Some(rook) = Castles::rook(wt) else {
                return false;
            };

            let threats = self.threats()[!turn].to_simd();
            return path & threats.simd_ne(zeroed()) == zeroed()
                && Bitboard::segment(ksq, rook.whence()) & occ == zeroed()
                && self.castles().has(wt);
        }

        if piece.role() == Pawn && !m.is_capture() {
            let third = Third.bitboard();
            let single = wc.bitboard().perspective(turn).shl(8).perspective(turn) & !occ;
            let double = (single.perspective(turn) & third).shl(8).perspective(turn) & !occ;
            if !single.contains(wt) & !double.contains(wt) {
                return false;
            }
        } else if !self.threats()[turn][wt].contains(self[wc].idx().assume()) {
            return false;
        }

        if piece.role() != King {
            let checks = self.checkers().iter().fold(Bitboard::empty(), |bb, idx| {
                let sq = self.squares()[!turn][idx].assume();
                Bitboard::segment(ksq, sq).with(sq) | bb
            });

            return checks.is_empty() || checks.contains(wt);
        }

        if !self.threats()[!turn][wt].is_empty() {
            return false;
        }

        for idx in self.checkers() {
            let checker = self.roles()[!turn][idx].assume();
            if matches!(checker, Role::Bishop | Role::Rook | Role::Queen) {
                let sq = self.squares()[!turn][idx].assume();
                if Bitboard::line(wc, sq).without(sq).contains(wt) {
                    return false;
                }
            }
        }

        true
    }

    /// Play a [`Move`].
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn play(&mut self, m: Move) {
        debug_assert!(self.is_legal(m), "{self} {m}");

        use Role::*;
        let wc = m.whence();
        let wt = m.whither();
        let src = self[wc];

        if src.role() == Some(Pawn) || m.is_noisy() {
            self.board.halfmoves = 0;
            self.history = zeroed();
        } else {
            let turn = self.turn();
            let hm = self.board.halfmoves as usize;
            self.board.halfmoves += 1;
            let entries = self.history[turn].len();
            self.history[turn][hm / 2 % entries] = self.zobrists().hash.cast();
        }

        if self.turn() == Color::Black {
            self.board.fullmoves += 1;
        }

        self.board.turn = !self.board.turn;
        self.zobrists.hash ^= ZobristNumbers::turn();
        if let Some(ep) = self.board.en_passant.take() {
            self.zobrists.hash ^= ZobristNumbers::en_passant(ep.file());
        }

        let victim = self[wt];
        let dst = m.promotion().map_or(src, |promotion| {
            Place::new(Piece::new(promotion, !self.turn()), src.idx().assume())
        });

        if !victim.is_empty() {
            self.zobrists.xor(wt, victim.piece().assume());
        } else if m.is_capture() {
            let sq = Square::new(wt.file(), wc.rank());
            self.zobrists.xor(sq, Piece::new(Role::Pawn, self.turn()));
            self.threats.outplace(&self.board, self[sq], sq);
            self.board.outplace(sq);
        } else if src.role() == Some(Pawn) && (wt - wc).abs() == 16 {
            let ep = Square::new(wc.file(), Rank::Third.perspective(!self.turn()));
            let theirs = self.board.roles()[self.turn()].matching(Some(Pawn));
            if self.threats()[self.turn()][ep] & theirs != zeroed() {
                self.zobrists.hash ^= ZobristNumbers::en_passant(ep.file());
                self.board.en_passant = Some(ep);
            }
        } else if src.role() == Some(King) && (wt - wc).abs() == 2 {
            #[inline(never)]
            fn play_castling(pos: &mut Position, sq: Square) {
                let castling = Castles::rook(sq).assume();
                let (wc, wt) = (castling.whence(), castling.whither());

                pos.zobrists.xor(wc, Piece::new(Rook, !pos.turn()));
                pos.zobrists.xor(wt, Piece::new(Rook, !pos.turn()));

                let rook = pos[wc];
                pos.threats.displace(&pos.board, rook, wc, rook, wt);
                pos.board.displace(wc, wt, rook);
            }

            play_castling(self, wt);
        }

        if victim.is_empty() {
            self.threats.displace(&self.board, src, wc, dst, wt);
        } else {
            self.threats.replace(&self.board, src, wc, dst, wt, victim);
            self.board.outplace(wt);
        }

        self.board.displace(wc, wt, dst);
        self.zobrists.xor(wc, src.piece().assume());
        self.zobrists.xor(wt, dst.piece().assume());
        self.direct_checks = self.board.direct_checks();
        self.pins = self.board.pins(self.threats());

        let disrupted = Castles::from(wc) | Castles::from(wt);
        if self.castles() & disrupted != Castles::none() {
            self.zobrists.hash ^= ZobristNumbers::castling(self.castles());
            self.board.castles &= !disrupted;
            self.zobrists.hash ^= ZobristNumbers::castling(self.castles());
        }
    }

    /// Play a null-move.
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pass(&mut self) {
        debug_assert!(!self.is_check());

        let turn = self.turn();
        if turn == Color::Black {
            self.board.fullmoves += 1;
        }

        let hm = self.board.halfmoves as usize;
        self.board.halfmoves += 1;
        let entries = self.history[turn].len();
        self.history[turn][hm / 2 % entries] = self.zobrists().hash.cast();

        self.board.turn = !self.board.turn;
        self.zobrists.hash ^= ZobristNumbers::turn();
        if let Some(ep) = self.board.en_passant.take() {
            self.zobrists.hash ^= ZobristNumbers::en_passant(ep.file());
        }

        self.pins = self.board.pins(self.threats());
    }

    /// Counts the total number of reachable positions to the given depth.
    pub fn perft(&self, depth: u8) -> u64 {
        match depth {
            0 => 1,
            1 => self.moves().len() as u64,
            _ => self
                .moves()
                .into_iter()
                .map(|m| {
                    let mut next = *self;
                    next.play(m);
                    next.perft(depth - 1)
                })
                .sum(),
        }
    }
}

impl From<Position> for Board {
    #[inline(always)]
    fn from(pos: Position) -> Self {
        pos.board
    }
}

impl Display for Position {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.board, f)
    }
}

/// The reason why parsing the FEN string failed.
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Error, From)]
pub enum ParsePositionError {
    #[display("failed to parse position")]
    InvalidFen(ParseFenError),
    #[display("illegal position")]
    IllegalPosition,
}

impl FromStr for Position {
    type Err = ParsePositionError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let board: Board = s.parse()?;
        let threats = board.threats();
        for color in Color::iter() {
            use ParsePositionError::IllegalPosition;
            board.king(color).ok_or(IllegalPosition)?;
        }

        Ok(Position {
            pins: board.pins(&threats),
            threats,
            zobrists: board.zobrists(),
            direct_checks: board.direct_checks(),
            history: zeroed(),
            board,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::sample::select;
    use std::{cmp::Reverse, collections::HashSet, fmt::Debug, hash::DefaultHasher};
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Position>>(), size_of::<Position>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn position_compares_by_board(a: Position, b: Position) {
        assert_eq!(a == b, a.board == b.board);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn hash_is_consistent(a: Position, b: Position) {
        let mut hasher = DefaultHasher::default();
        a.hash(&mut hasher);
        let x = hasher.finish();

        let mut hasher = DefaultHasher::default();
        b.hash(&mut hasher);
        let y = hasher.finish();

        assert!(a != b || x == y);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn occupied_returns_non_empty_places(pos: Position) {
        for sq in Bitboard::from(pos.occupied()) {
            assert_ne!(pos[sq], Place::empty());
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn king_returns_square_occupied_by_a_king(pos: Position, c: Color) {
        assert_eq!(pos[pos.king(c)].piece(), Some(Piece::new(Role::King, c)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn zobrist_hashes_are_updated_incrementally(pos: Position) {
        assert_eq!(pos.zobrists, pos.board.zobrists());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn threats_are_updated_incrementally(pos: Position) {
        assert_eq!(pos.threats, pos.board.threats());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn checkmate_implies_outcome(pos: Position) {
        assert!(!pos.is_checkmate() || pos.outcome() == Some(Outcome::Checkmate(!pos.turn())));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn stalemate_implies_outcome(pos: Position) {
        assert!(!pos.is_stalemate() || pos.outcome() == Some(Outcome::Stalemate));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn checkmate_implies_check(pos: Position) {
        assert!(!pos.is_checkmate() || pos.is_check());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn checkmate_and_stalemate_are_mutually_exclusive(pos: Position) {
        assert!(!(pos.is_checkmate() && pos.is_stalemate()));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn checkmate_implies_no_legal_move(pos: Position) {
        assert!(!pos.is_checkmate() || pos.moves().is_empty());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn stalemate_implies_no_legal_move(pos: Position) {
        assert!(!pos.is_stalemate() || pos.moves().is_empty());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn check_and_stalemate_are_mutually_exclusive(pos: Position) {
        assert!(!(pos.is_check() && pos.is_stalemate()));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn moves_returns_all_legal_moves(#[filter(#pos.outcome().is_none())] pos: Position) {
        for m in pos.moves() {
            assert!(pos.is_legal(m));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn moves_to_returns_legal_moves_to_a_set_of_squares(
        #[filter(#pos.outcome().is_none())] pos: Position,
        bb: Bitboard,
    ) {
        let a = pos.moves_to(bb);
        let b = Moves::from_iter(pos.moves().into_iter().filter(|m| bb.contains(m.whither())));

        assert_eq!(a.len(), b.len());
        assert_eq!(HashSet::<_>::from_iter(a), HashSet::from_iter(b));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn noisy_returns_legal_noisy_moves(#[filter(#pos.outcome().is_none())] pos: Position) {
        let a = pos.noisy();
        let b = Moves::from_iter(pos.moves().into_iter().filter(|m| m.is_noisy()));

        assert_eq!(a.len(), b.len());
        assert_eq!(HashSet::<_>::from_iter(a), HashSet::from_iter(b));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn noisy_to_returns_legal_noisy_moves_to_a_set_of_squares(
        #[filter(#pos.outcome().is_none())] pos: Position,
        bb: Bitboard,
    ) {
        let a = pos.noisy_to(bb);
        let b = Moves::from_iter(pos.noisy().into_iter().filter(|m| bb.contains(m.whither())));

        assert_eq!(a.len(), b.len());
        assert_eq!(HashSet::<_>::from_iter(a), HashSet::from_iter(b));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn exchanges_iterator_is_sorted_by_captor_of_least_value(
        #[filter(#pos.outcome().is_none())] pos: Position,
        #[map(|s: Selector| s.select(#pos.moves()))] m: Move,
    ) {
        let sq = m.whither();
        let exchanges = pos.exchanges(m);
        let mut pos = pos;
        pos.play(m);

        for (m, captor) in exchanges {
            assert_eq!(
                Some((Some(captor), m.promotion())),
                pos.noisy_to(sq.bitboard())
                    .into_iter()
                    .map(|m| (pos[m.whence()].role(), m.promotion()))
                    .min_by_key(|&(r, p)| (r, Reverse(p)))
            );

            pos.play(m);
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn captures_reduce_material(
        #[filter(#pos.noisy().into_iter().any(Move::is_capture))] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.noisy().into_iter().filter(|m| m.is_capture())))] m: Move,
    ) {
        let prev = pos;
        pos.play(m);
        assert!(pos.by_color(pos.turn()).count() < prev.by_color(pos.turn()).count());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn promotions_exchange_pawns(
        #[filter(#pos.noisy().into_iter().any(Move::is_promotion))] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.noisy().into_iter().filter(|m| m.is_promotion())))]
        m: Move,
    ) {
        let prev = pos;
        pos.play(m);

        assert!(pos.pawns(prev.turn()).count() < prev.pawns(prev.turn()).count());

        assert_eq!(
            pos.by_color(prev.turn()).count(),
            prev.by_color(prev.turn()).count()
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn move_is_legal_if_can_be_played(#[filter(#pos.outcome().is_none())] pos: Position, m: Move) {
        assert_eq!(pos.is_legal(m), pos.moves().into_iter().any(|n| m == n));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn move_gives_direct_check_if_threatens_opposing_king_directly(
        #[filter(#pos.outcome().is_none())] pos: Position,
        #[map(|s: Selector| s.select(#pos.moves()))] m: Move,
    ) {
        let mut next = pos;
        next.play(m);
        assert!(!pos.gives_direct_check(m) || next.is_check());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn legal_move_updates_position(
        #[filter(#pos.outcome().is_none())] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves()))] m: Move,
    ) {
        let prev = pos;
        pos.play(m);

        assert_ne!(pos, prev);
        assert_ne!(pos.turn(), prev.turn());

        assert_eq!(pos[m.whence()], Place::empty());
        assert_eq!(
            pos[m.whither()].piece(),
            m.promotion()
                .map(|r| Piece::new(r, prev.turn()))
                .or_else(|| prev[m.whence()].piece())
        );

        assert_eq!(
            Bitboard::from(pos.occupied()),
            Role::iter().fold(Bitboard::empty(), |bb, r| bb | pos.by_role(r))
        );

        for r in Role::iter() {
            for sq in Role::iter() {
                if r != sq {
                    assert_eq!(Bitboard::from(pos.by_role(r) & pos.by_role(sq)), zeroed());
                }
            }
        }

        assert_eq!(
            pos.by_color(prev.turn()).count(),
            prev.by_color(prev.turn()).count()
        );

        assert_eq!(
            pos.by_color(pos.turn()).count(),
            prev.by_color(pos.turn()).count() - m.is_capture() as u32
        );

        if let Some(ep) = pos.en_passant() {
            assert_eq!(ep.rank(), Rank::Sixth.perspective(pos.turn()));
        }
    }

    #[proptest]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn play_panics_if_move_illegal(mut pos: Position, #[filter(!#pos.is_legal(#m))] m: Move) {
        pos.play(m);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn pass_updates_position(#[filter(!#pos.is_check())] mut pos: Position) {
        let prev = pos;
        pos.pass();
        assert_ne!(pos, prev);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn pass_reverts_itself(#[filter(!#pos.is_check() )] mut pos: Position) {
        let prev = pos;
        pos.pass();
        pos.pass();
        assert_eq!(pos.placement(), prev.placement());
        assert_eq!(pos.threats(), prev.threats());
        assert_eq!(pos.pins(), prev.pins());
    }

    #[proptest]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn pass_panics_if_in_check(#[filter(#pos.is_check())] mut pos: Position) {
        pos.pass();
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn threefold_repetition_implies_draw(#[filter(#pos.outcome().is_none() )] mut pos: Position) {
        let zobrist = pos.zobrists().hash.cast();
        prop_assume!(zobrist != zeroed());

        let turn = pos.turn();
        pos.history[turn][..2].clone_from_slice(&[zobrist, zobrist]);
        assert!(pos.is_draw_by_repetition());
        assert_eq!(pos.outcome(), Some(Outcome::DrawByThreefoldRepetition));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_position_is_an_identity(pos: Position) {
        assert_eq!(pos.to_string().parse(), Ok(pos));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    #[expect(clippy::string_slice)]
    fn parsing_position_fails_for_invalid_fen(
        pos: Position,
        #[strategy(..=#pos.to_string().len())] n: usize,
        #[strategy("[^[:ascii:]]+")] r: String,
    ) {
        let s = pos.to_string();

        assert_eq!(
            [&s[..n], &r, &s[n..]].concat().parse().ok(),
            None::<Position>
        );
    }

    #[rustfmt::skip]
    #[cfg(not(coverage))]
    const PERFT_SUITE: &[(&str, u8, u64)] = &[
        ("1r2k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1", 6, 195629489),
        ("1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1", 6, 198328929),
        ("2r1k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1", 6, 184411439),
        ("2r1k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1", 6, 185959088),
        ("3k4/3pp3/8/8/8/8/3PP3/3K4 b - - 0 1", 6, 199002),
        ("3k4/3pp3/8/8/8/8/3PP3/3K4 w - - 0 1", 6, 199002),
        ("4k2r/6K1/8/8/8/8/8/8 b k - 0 1", 6, 185867),
        ("4k2r/6K1/8/8/8/8/8/8 w k - 0 1", 6, 179869),
        ("4k2r/8/8/8/8/8/8/4K3 b k - 0 1", 6, 764643),
        ("4k2r/8/8/8/8/8/8/4K3 w k - 0 1", 6, 899442),
        ("4k3/4p3/4K3/8/8/8/8/8 b - - 0 1", 6, 11848),
        ("4k3/8/8/8/8/8/8/4K2R b K - 0 1", 6, 899442),
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", 6, 764643),
        ("4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1", 6, 3517770),
        ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", 6, 2788982),
        ("4k3/8/8/8/8/8/8/R3K3 b Q - 0 1", 6, 1001523),
        ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 846648),
        ("6KQ/8/8/8/8/8/8/7k b - - 0 1", 6, 391507),
        ("6kq/8/8/8/8/8/8/7K w - - 0 1", 6, 391507),
        ("6qk/8/8/8/8/8/8/7K b - - 0 1", 6, 419369),
        ("7k/3p4/8/8/3P4/8/8/K7 b - - 0 1", 6, 32167),
        ("7k/3p4/8/8/3P4/8/8/K7 w - - 0 1", 6, 32191),
        ("7K/7p/7k/8/8/8/8/8 b - - 0 1", 6, 6249),
        ("7K/7p/7k/8/8/8/8/8 w - - 0 1", 6, 2343),
        ("7k/8/1p6/8/8/P7/8/7K b - - 0 1", 6, 29679),
        ("7k/8/1p6/8/8/P7/8/7K w - - 0 1", 6, 29679),
        ("7k/8/8/1p6/P7/8/8/7K b - - 0 1", 6, 41874),
        ("7k/8/8/1p6/P7/8/8/7K w - - 0 1", 6, 41874),
        ("7k/8/8/3p4/8/8/3P4/K7 b - - 0 1", 6, 30749),
        ("7k/8/8/3p4/8/8/3P4/K7 w - - 0 1", 6, 30980),
        ("7k/8/8/p7/1P6/8/8/7K b - - 0 1", 6, 41874),
        ("7k/8/8/p7/1P6/8/8/7K w - - 0 1", 6, 41874),
        ("7k/8/p7/8/8/1P6/8/7K b - - 0 1", 6, 29679),
        ("7k/8/p7/8/8/1P6/8/7K w - - 0 1", 6, 29679),
        ("7k/RR6/8/8/8/8/rr6/7K b - - 0 1", 6, 44956585),
        ("7k/RR6/8/8/8/8/rr6/7K w - - 0 1", 6, 44956585),
        ("8/1k6/8/5N2/8/4n3/8/2K5 b - - 0 1", 6, 3147566),
        ("8/1k6/8/5N2/8/4n3/8/2K5 w - - 0 1", 6, 2594412),
        ("8/1n4N1/2k5/8/8/5K2/1N4n1/8 b - - 0 1", 6, 8503277),
        ("8/1n4N1/2k5/8/8/5K2/1N4n1/8 w - - 0 1", 6, 8107539),
        ("8/2k1p3/3pP3/3P2K1/8/8/8/8 b - - 0 1", 6, 34822),
        ("8/2k1p3/3pP3/3P2K1/8/8/8/8 w - - 0 1", 6, 34834),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
        ("8/3k4/3p4/8/3P4/3K4/8/8 b - - 0 1", 6, 158065),
        ("8/3k4/3p4/8/3P4/3K4/8/8 w - - 0 1", 6, 157093),
        ("8/8/1B6/7b/7k/8/2B1b3/7K b - - 0 1", 6, 29027891),
        ("8/8/1B6/7b/7k/8/2B1b3/7K w - - 0 1", 6, 28861171),
        ("8/8/3K4/3Nn3/3nN3/4k3/8/8 b - - 0 1", 6, 4405103),
        ("8/8/3k4/3p4/3P4/3K4/8/8 b - - 0 1", 6, 53138),
        ("8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1", 6, 53138),
        ("8/8/3k4/3p4/8/3P4/3K4/8 b - - 0 1", 6, 157093),
        ("8/8/3k4/3p4/8/3P4/3K4/8 w - - 0 1", 6, 158065),
        ("8/8/4k3/3Nn3/3nN3/4K3/8/8 w - - 0 1", 6, 19870403),
        ("8/8/7k/7p/7P/7K/8/8 b - - 0 1", 6, 10724),
        ("8/8/7k/7p/7P/7K/8/8 w - - 0 1", 6, 10724),
        ("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1", 6, 11848),
        ("8/8/8/8/8/7K/7P/7k b - - 0 1", 6, 2343),
        ("8/8/8/8/8/7K/7P/7k w - - 0 1", 6, 6249),
        ("8/8/8/8/8/8/1k6/R3K3 b Q - 0 1", 6, 367724),
        ("8/8/8/8/8/8/1k6/R3K3 w Q - 0 1", 6, 413018),
        ("8/8/8/8/8/8/6k1/4K2R b K - 0 1", 6, 179869),
        ("8/8/8/8/8/8/6k1/4K2R w K - 0 1", 6, 185867),
        ("8/8/8/8/8/K7/P7/k7 b - - 0 1", 6, 2343),
        ("8/8/8/8/8/K7/P7/k7 w - - 0 1", 6, 6249),
        ("8/8/k7/p7/P7/K7/8/8 b - - 0 1", 6, 10724),
        ("8/8/k7/p7/P7/K7/8/8 w - - 0 1", 6, 10724),
        ("8/Pk6/8/8/8/8/6Kp/8 b - - 0 1", 6, 1030499),
        ("8/Pk6/8/8/8/8/6Kp/8 w - - 0 1", 6, 1030499),
        ("8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1", 6, 28859283),
        ("8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1", 6, 28859283),
        ("B6b/8/8/8/2K5/4k3/8/b6B w - - 0 1", 6, 22823890),
        ("B6b/8/8/8/2K5/5k2/8/b6B b - - 0 1", 6, 9250746),
        ("k7/6p1/8/8/8/8/7P/K7 b - - 0 1", 6, 55338),
        ("k7/6p1/8/8/8/8/7P/K7 w - - 0 1", 6, 55338),
        ("k7/7p/8/8/8/8/6P1/K7 b - - 0 1", 6, 55338),
        ("k7/7p/8/8/8/8/6P1/K7 w - - 0 1", 6, 55338),
        ("k7/8/2N5/1N6/8/8/8/K6n b - - 0 1", 6, 588695),
        ("K7/8/2n5/1n6/8/8/8/k6N b - - 0 1", 6, 688780),
        ("K7/8/2n5/1n6/8/8/8/k6N w - - 0 1", 6, 588695),
        ("k7/8/2N5/1N6/8/8/8/K6n w - - 0 1", 6, 688780),
        ("k7/8/3p4/8/3P4/8/8/7K b - - 0 1", 6, 21104),
        ("k7/8/3p4/8/3P4/8/8/7K w - - 0 1", 6, 20960),
        ("k7/8/3p4/8/8/4P3/8/7K b - - 0 1", 6, 28662),
        ("k7/8/3p4/8/8/4P3/8/7K w - - 0 1", 6, 28662),
        ("k7/8/6p1/8/8/7P/8/K7 b - - 0 1", 6, 29679),
        ("k7/8/6p1/8/8/7P/8/K7 w - - 0 1", 6, 29679),
        ("k7/8/7p/8/8/6P1/8/K7 b - - 0 1", 6, 29679),
        ("k7/8/7p/8/8/6P1/8/K7 w - - 0 1", 6, 29679),
        ("k7/8/8/3p4/4p3/8/8/7K b - - 0 1", 6, 22579),
        ("k7/8/8/3p4/4p3/8/8/7K w - - 0 1", 6, 22886),
        ("K7/8/8/3Q4/4q3/8/8/7k b - - 0 1", 6, 3370175),
        ("K7/8/8/3Q4/4q3/8/8/7k w - - 0 1", 6, 3370175),
        ("k7/8/8/6p1/7P/8/8/K7 b - - 0 1", 6, 41874),
        ("k7/8/8/6p1/7P/8/8/K7 w - - 0 1", 6, 41874),
        ("k7/8/8/7p/6P1/8/8/K7 b - - 0 1", 6, 41874),
        ("k7/8/8/7p/6P1/8/8/K7 w - - 0 1", 6, 41874),
        ("k7/B7/1B6/1B6/8/8/8/K6b b - - 0 1", 6, 7382896),
        ("K7/b7/1b6/1b6/8/8/8/k6B b - - 0 1", 6, 7881673),
        ("K7/b7/1b6/1b6/8/8/8/k6B w - - 0 1", 6, 7382896),
        ("k7/B7/1B6/1B6/8/8/8/K6b w - - 0 1", 6, 7881673),
        ("K7/p7/k7/8/8/8/8/8 b - - 0 1", 6, 6249),
        ("K7/p7/k7/8/8/8/8/8 w - - 0 1", 6, 2343),
        ("n1n5/1Pk5/8/8/8/8/5Kp1/5N1N b - - 0 1", 6, 37665329),
        ("n1n5/1Pk5/8/8/8/8/5Kp1/5N1N w - - 0 1", 6, 37665329),
        ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 6, 71179139),
        ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1", 6, 71179139),
        ("r3k1r1/8/8/8/8/8/8/R3K2R b KQq - 0 1", 6, 189224276),
        ("r3k1r1/8/8/8/8/8/8/R3K2R w KQq - 0 1", 6, 190755813),
        ("r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 0 1", 6, 198328929),
        ("r3k2r/8/8/8/8/8/8/1R2K2R w Kkq - 0 1", 6, 195629489),
        ("r3k2r/8/8/8/8/8/8/2R1K2R b Kkq - 0 1", 6, 185959088),
        ("r3k2r/8/8/8/8/8/8/2R1K2R w Kkq - 0 1", 6, 184411439),
        ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", 6, 2788982),
        ("r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1", 6, 3517770),
        ("r3k2r/8/8/8/8/8/8/R3K1R1 b Qkq - 0 1", 6, 190755813),
        ("r3k2r/8/8/8/8/8/8/R3K1R1 w Qkq - 0 1", 6, 189224276),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 6, 179862938),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 6, 179862938),
        ("r3k3/1K6/8/8/8/8/8/8 b q - 0 1", 6, 413018),
        ("r3k3/1K6/8/8/8/8/8/8 w q - 0 1", 6, 367724),
        ("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", 6, 846648),
        ("r3k3/8/8/8/8/8/8/4K3 w q - 0 1", 6, 1001523),
        ("R6r/8/8/2K5/5k2/8/8/r6R b - - 0 1", 6, 524966748),
        ("R6r/8/8/2K5/5k2/8/8/r6R w - - 0 1", 6, 525169084),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("rnbqkb1r/ppppp1pp/7n/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 5, 11139762),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
    ];

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn perft_counts_all_reachable_positions_up_to_ply(
        #[strategy(select(PERFT_SUITE))] entry: (&'static str, u8, u64),
    ) {
        let (fen, plies, count) = entry;
        let pos: Position = fen.parse()?;
        assert_eq!(pos.perft(plies.saturate()), count);
    }
}
