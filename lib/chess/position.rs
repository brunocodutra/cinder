use crate::util::{Assume, Integer};
use crate::{chess::*, search::Depth};
use arrayvec::ArrayVec;
use derive_more::with_trait::{Debug, Deref, DerefMut, Display, Error, From, IntoIterator};
use std::fmt::{self, Formatter};
use std::hash::{Hash, Hasher};
use std::{num::NonZeroU32, str::FromStr};

#[cfg(test)]
use proptest::{prelude::*, sample::*};

/// A container with sufficient capacity to hold all [`Move`]s in any [`Position`].
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Deref, DerefMut, IntoIterator)]
pub struct MovePack(ArrayVec<MoveSet, 32>);

impl MovePack {
    #[inline(always)]
    pub fn unpack(&self) -> impl Iterator<Item = Move> {
        self.unpack_if(|_| true)
    }

    #[inline(always)]
    pub fn unpack_if<F: FnMut(&MoveSet) -> bool>(&self, f: F) -> impl Iterator<Item = Move> {
        self.iter().copied().filter(f).flatten()
    }
}

/// The [`MovePacker`] is out of capacity.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error, From)]
struct CapacityError;

trait MovePacker {
    fn pack(
        &mut self,
        piece: Piece,
        wc: Square,
        wt: Bitboard,
        victims: Bitboard,
    ) -> Result<(), CapacityError>;
}

struct NoCapacityMovePacker;

impl MovePacker for NoCapacityMovePacker {
    #[inline(always)]
    fn pack(
        &mut self,
        _: Piece,
        _: Square,
        wt: Bitboard,
        _: Bitboard,
    ) -> Result<(), CapacityError> {
        if wt == Bitboard::empty() {
            Ok(())
        } else {
            Err(CapacityError)
        }
    }
}

impl MovePacker for MovePack {
    #[inline(always)]
    fn pack(
        &mut self,
        piece: Piece,
        wc: Square,
        wt: Bitboard,
        victims: Bitboard,
    ) -> Result<(), CapacityError> {
        let captures = wt & victims;
        let regulars = wt & !victims;

        if !captures.is_empty() {
            let captures = MoveSet::capture(piece, wc, captures);
            self.0.try_push(captures).assume();
        }

        if !regulars.is_empty() {
            let regulars = MoveSet::regular(piece, wc, regulars);
            self.0.try_push(regulars).assume();
        }

        Ok(())
    }
}

enum EvasionGenerator {}

impl EvasionGenerator {
    #[inline(always)]
    fn generate<T: MovePacker>(pos: &Position, packer: &mut T) -> Result<(), CapacityError> {
        let turn = pos.turn();
        let ours = pos.material(turn);
        let theirs = pos.material(!turn);
        let occupied = ours | theirs;
        let king = pos.king(turn);

        let checks = pos.checkers().iter().fold(Bitboard::empty(), |bb, sq| {
            Bitboard::segment(king, sq).union(bb)
        });

        let candidates = match pos.checkers().len() {
            1 => ours & !pos.pinned(),
            _ => king.bitboard(),
        };

        for wc in candidates & pos.by_role(Role::Pawn) {
            let piece = Piece::new(Role::Pawn, turn);
            let ep = pos.en_passant().map_or(Bitboard::empty(), Square::bitboard);
            let mut moves = piece.moves(wc, ours, theirs) & checks;
            moves |= piece.attacks(wc, occupied) & (pos.checkers() | ep);

            for wt in moves & ep {
                let target = Square::new(wt.file(), wc.rank());
                let blockers = occupied.without(target).without(wc);
                if pos.is_threatened(king, !turn, blockers) {
                    moves ^= ep;
                }
            }

            packer.pack(piece, wc, moves, theirs | ep)?;
        }

        {
            let piece = Piece::new(Role::Knight, turn);
            for wc in candidates & pos.by_role(Role::Knight) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::Bishop, turn);
            for wc in candidates & pos.by_role(Role::Bishop) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::Rook, turn);
            for wc in candidates & pos.by_role(Role::Rook) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::Queen, turn);
            for wc in candidates & pos.by_role(Role::Queen) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::King, turn);
            let mut moves = piece.moves(king, ours, theirs) & !checks;
            for wt in moves {
                if pos.is_threatened(wt, !turn, occupied.without(king)) {
                    moves ^= wt.bitboard();
                }
            }

            packer.pack(piece, king, moves, theirs)?;
        }

        Ok(())
    }
}

enum MoveGenerator {}

impl MoveGenerator {
    #[inline(always)]
    fn generate<T: MovePacker>(pos: &Position, packer: &mut T) -> Result<(), CapacityError> {
        let turn = pos.turn();
        let ours = pos.material(turn);
        let theirs = pos.material(!turn);
        let occupied = ours | theirs;
        let king = pos.king(turn);

        for wc in ours & pos.by_role(Role::Pawn) {
            let piece = Piece::new(Role::Pawn, turn);
            let ep = pos.en_passant().map_or(Bitboard::empty(), Square::bitboard);
            let mut moves = piece.moves(wc, ours, theirs);
            moves |= piece.attacks(wc, occupied) & (theirs | ep);
            if pos.pinned().contains(wc) {
                moves &= Bitboard::line(king, wc);
            }

            for wt in moves & ep {
                let target = Square::new(wt.file(), wc.rank());
                let blockers = occupied.without(target).without(wc).with(wt);
                if pos.is_threatened(king, !turn, blockers) {
                    moves ^= ep;
                }
            }

            packer.pack(piece, wc, moves, theirs | ep)?;
        }

        {
            let piece = Piece::new(Role::Knight, turn);
            for wc in ours & pos.by_role(Role::Knight) & !pos.pinned() {
                let moves = piece.moves(wc, ours, theirs);
                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::Bishop, turn);
            for wc in ours & pos.by_role(Role::Bishop) {
                let mut moves = piece.moves(wc, ours, theirs);
                if pos.pinned().contains(wc) {
                    moves &= Bitboard::line(king, wc);
                }

                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::Rook, turn);
            for wc in ours & pos.by_role(Role::Rook) {
                let mut moves = piece.moves(wc, ours, theirs);
                if pos.pinned().contains(wc) {
                    moves &= Bitboard::line(king, wc);
                }

                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::Queen, turn);
            for wc in ours & pos.by_role(Role::Queen) {
                let mut moves = piece.moves(wc, ours, theirs);
                if pos.pinned().contains(wc) {
                    moves &= Bitboard::line(king, wc);
                }

                packer.pack(piece, wc, moves, theirs)?;
            }
        }

        {
            let piece = Piece::new(Role::King, turn);
            let mut moves = piece.moves(king, ours, theirs);
            for wt in moves {
                if pos.is_threatened(wt, !turn, occupied) {
                    moves ^= wt.bitboard();
                }
            }

            if let Some(c) = pos.castles().long(turn) {
                let b = Square::new(File::B, c.rank());
                let path = c.bitboard().with(Square::new(File::D, c.rank()));
                if occupied & path.with(b) == Bitboard::empty() {
                    if !path.iter().any(|sq| pos.is_threatened(sq, !turn, occupied)) {
                        moves |= c.bitboard();
                    }
                }
            }

            if let Some(g) = pos.castles().short(turn) {
                let path = g.bitboard().with(Square::new(File::F, g.rank()));
                if occupied & path == Bitboard::empty() {
                    if !path.iter().any(|sq| pos.is_threatened(sq, !turn, occupied)) {
                        moves |= g.bitboard();
                    }
                }
            }

            packer.pack(piece, king, moves, theirs)?;
        }

        Ok(())
    }
}

/// The current position on the board.
///
/// This type guarantees that it only holds valid positions.
#[derive(Debug, Clone, Eq)]
#[debug("Position({self})")]
pub struct Position {
    board: Board,
    zobrist: Zobrist,
    checkers: Bitboard,
    pinned: Bitboard,
    history: [[Option<NonZeroU32>; 32]; 2],
}

#[cfg(test)]
impl Arbitrary for Position {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (0..256, any::<Selector>())
            .prop_map(|(moves, selector)| {
                let mut pos = Position::default();

                for _ in 0..moves {
                    if pos.outcome().is_none() {
                        pos.play(selector.select(pos.moves().unpack()));
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
    #[inline(always)]
    fn default() -> Self {
        let board = Board::default();

        Self {
            zobrist: board.zobrist(),
            checkers: Default::default(),
            pinned: Default::default(),
            history: Default::default(),
            board,
        }
    }
}

impl Hash for Position {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.board.hash(state);
    }
}

impl PartialEq for Position {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.board.eq(&other.board)
    }
}

impl Position {
    /// The side to move.
    #[inline(always)]
    pub fn turn(&self) -> Color {
        self.board.turn
    }

    /// The number of halfmoves since the last capture or pawn advance.
    ///
    /// It resets to 0 whenever a piece is captured or a pawn is moved.
    #[inline(always)]
    pub fn halfmoves(&self) -> u8 {
        self.board.halfmoves
    }

    /// The current move number since the start of the game.
    ///
    /// It starts at 1, and is incremented after every move by black.
    #[inline(always)]
    pub fn fullmoves(&self) -> NonZeroU32 {
        self.board.fullmoves.convert().assume()
    }

    /// The en passant square.
    #[inline(always)]
    pub fn en_passant(&self) -> Option<Square> {
        self.board.en_passant
    }

    /// The castle rights.
    #[inline(always)]
    pub fn castles(&self) -> Castles {
        self.board.castles
    }

    /// [`Square`]s occupied.
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.material(Color::White) ^ self.material(Color::Black)
    }

    /// [`Square`]s occupied by pieces of a [`Color`].
    #[inline(always)]
    pub fn material(&self, side: Color) -> Bitboard {
        self.board.material(side)
    }

    /// [`Square`]s occupied by pieces of a [`Role`].
    #[inline(always)]
    pub fn by_role(&self, role: Role) -> Bitboard {
        self.board.by_role(role)
    }

    /// [`Square`]s occupied by a [`Piece`].
    #[inline(always)]
    pub fn by_piece(&self, piece: Piece) -> Bitboard {
        self.board.by_piece(piece)
    }

    /// [`Square`]s occupied by pawns of a [`Color`].
    #[inline(always)]
    pub fn pawns(&self, side: Color) -> Bitboard {
        self.by_piece(Piece::new(Role::Pawn, side))
    }

    /// [`Square`]s occupied by pieces other than pawns of a [`Color`].
    #[inline(always)]
    pub fn pieces(&self, side: Color) -> Bitboard {
        self.material(side) ^ self.pawns(side)
    }

    /// [`Square`] occupied by a the king of a [`Color`].
    #[inline(always)]
    pub fn king(&self, side: Color) -> Square {
        self.board.king(side).assume()
    }

    /// The [`Color`] of the piece on the given [`Square`], if any.
    #[inline(always)]
    pub fn color_on(&self, sq: Square) -> Option<Color> {
        self.board.color_on(sq)
    }

    /// The [`Role`] of the piece on the given [`Square`], if any.
    #[inline(always)]
    pub fn role_on(&self, sq: Square) -> Option<Role> {
        self.board.role_on(sq)
    }

    /// The [`Piece`] on the given [`Square`], if any.
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.board.piece_on(sq)
    }

    /// An iterator over all pieces on the board.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (Piece, Square)> {
        self.board.iter()
    }

    /// This position's [zobrist hash].
    ///
    /// [zobrist hash]: https://www.chessprogramming.org/Zobrist_Hashing
    #[inline(always)]
    pub fn zobrist(&self) -> Zobrist {
        self.zobrist
    }

    /// [`Square`]s occupied by pieces giving check.
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.checkers
    }

    /// [`Square`]s occupied by pieces pinned.
    #[inline(always)]
    pub fn pinned(&self) -> Bitboard {
        self.pinned
    }

    /// How many other times this position has repeated.
    #[inline(always)]
    pub fn repetitions(&self) -> usize {
        match NonZeroU32::new(self.zobrist().cast()) {
            None => 0,
            hash => {
                let history = &self.history[self.turn() as usize];
                history.iter().filter(|h| **h == hash).count()
            }
        }
    }

    /// Whether a [`Square`] is threatened by a piece of a [`Color`].
    #[inline(always)]
    pub fn is_threatened(&self, sq: Square, side: Color, occupied: Bitboard) -> bool {
        let theirs = self.material(side);
        for role in [Role::Pawn, Role::Knight, Role::King] {
            let candidates = occupied & theirs & self.by_role(role);
            if Piece::new(role, !side).attacks(sq, occupied) & candidates != Bitboard::empty() {
                return true;
            }
        }

        let queens = self.by_role(Role::Queen);
        for role in [Role::Bishop, Role::Rook] {
            let candidates = occupied & theirs & (queens | self.by_role(role));
            if Piece::new(role, !side).attacks(sq, occupied) & candidates != Bitboard::empty() {
                return true;
            }
        }

        false
    }

    /// Whether this position is a [check].
    ///
    /// [check]: https://www.chessprogramming.org/Check
    #[inline(always)]
    pub fn is_check(&self) -> bool {
        !self.checkers().is_empty()
    }

    /// Whether this position is a [checkmate].
    ///
    /// [checkmate]: https://www.chessprogramming.org/Checkmate
    #[inline(always)]
    pub fn is_checkmate(&self) -> bool {
        self.is_check() && EvasionGenerator::generate(self, &mut NoCapacityMovePacker).is_ok()
    }

    /// Whether this position is a [stalemate].
    ///
    /// [stalemate]: https://www.chessprogramming.org/Stalemate
    #[inline(always)]
    pub fn is_stalemate(&self) -> bool {
        !self.is_check() && MoveGenerator::generate(self, &mut NoCapacityMovePacker).is_ok()
    }

    /// Whether the game is a draw by [repetition].
    ///
    /// [repetition]: https://en.wikipedia.org/wiki/Threefold_repetition
    #[inline(always)]
    pub fn is_draw_by_repetition(&self) -> bool {
        self.repetitions() > 0
    }

    /// Whether the game is a draw by the [50-move rule].
    ///
    /// [50-move rule]: https://en.wikipedia.org/wiki/Fifty-move_rule
    #[inline(always)]
    pub fn is_draw_by_50_move_rule(&self) -> bool {
        self.halfmoves() >= 100
    }

    /// Whether this position has [insufficient material].
    ///
    /// [insufficient material]: https://www.chessprogramming.org/Material#InsufficientMaterial
    #[inline(always)]
    pub fn is_material_insufficient(&self) -> bool {
        use {Piece::*, Role::*};
        match self.occupied().len() {
            2 => true,
            3 => self.by_role(Bishop) | self.by_role(Knight) != Bitboard::empty(),
            4 => {
                let wb = self.by_piece(WhiteBishop);
                let bb = self.by_piece(BlackBishop);

                let dark = Bitboard::dark();
                let light = Bitboard::light();

                !(light.intersection(wb).is_empty() || light.intersection(bb).is_empty())
                    || !(dark.intersection(wb).is_empty() || dark.intersection(bb).is_empty())
            }
            _ => false,
        }
    }

    /// The [`Outcome`] of the game in case this position is final.
    #[inline(always)]
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

    /// The legal moves that can be played in this position.
    #[inline(always)]
    pub fn moves(&self) -> MovePack {
        let mut moves = MovePack::default();

        if self.is_check() {
            EvasionGenerator::generate(self, &mut moves).assume()
        } else {
            MoveGenerator::generate(self, &mut moves).assume()
        }

        moves
    }

    /// The sequence of captures om a square starting from a move ordered by least valued captor.
    ///
    /// Pins and checks are ignored.
    #[inline(always)]
    pub fn exchanges(&self, m: Move) -> impl Iterator<Item = (Move, Role, Role)> {
        use {Color::*, Piece::*, Role::*};

        gen move {
            let sq = m.whither();
            let queens = self.by_role(Queen);
            let rooks = self.by_role(Rook);
            let bishops = self.by_role(Bishop);

            let mut turn = self.turn();
            let mut attackers = Bitboard::empty();
            let mut occupied = self.occupied().without(m.whence()).without(sq);
            let mut victim = match m.promotion() {
                None => self.role_on(m.whence()).assume(),
                Some(r) => r,
            };

            for piece in [WhitePawn, BlackPawn] {
                attackers |= self.by_piece(piece) & piece.flip().attacks(sq, occupied);
            }

            for role in [Knight, King] {
                let candidates = self.by_role(role);
                attackers |= candidates & Piece::new(role, White).attacks(sq, occupied);
            }

            for (role, candidates) in [(Bishop, bishops | queens), (Rook, rooks | queens)] {
                attackers |= candidates & Piece::new(role, White).attacks(sq, occupied);
            }

            loop {
                turn = !turn;
                let candidates = attackers & self.material(turn);
                if candidates.is_empty() {
                    break;
                }

                let mut lva = None;
                for role in [Pawn, Knight, Bishop, Rook, Queen, King] {
                    let bb = candidates & self.by_role(role);
                    if let Some(wc) = bb.into_iter().next() {
                        let piece = Piece::new(role, turn);
                        let moves = MoveSet::capture(piece, wc, sq.bitboard());
                        lva = moves.into_iter().next().map(|m| (m, role));
                        occupied ^= wc.bitboard();
                        break;
                    }
                }

                let (m, captor) = lva.assume();
                if matches!(captor, Pawn | Bishop | Queen) {
                    attackers |= (bishops | queens) & WhiteBishop.attacks(sq, occupied);
                }

                if matches!(captor, Rook | Queen) {
                    attackers |= (rooks | queens) & WhiteRook.attacks(sq, occupied);
                }

                attackers &= occupied;
                if captor == King && !self.material(!turn).intersection(attackers).is_empty() {
                    break;
                }

                yield (m, captor, victim);
                victim = m.promotion().unwrap_or(captor);
            }
        }
    }

    /// Play a [`Move`].
    #[inline(always)]
    pub fn play(&mut self, m: Move) -> (Role, Option<(Role, Square)>) {
        debug_assert!(self.moves().unpack().any(|n| m == n));

        use {Role::*, Square::*};

        let turn = self.turn();
        let promotion = m.promotion();
        let (wc, wt) = (m.whence(), m.whither());
        let role = self.role_on(wc).assume();
        let capture = match self.role_on(wt) {
            _ if !m.is_capture() => None,
            Some(r) => Some((r, wt)),
            None => Some((Pawn, Square::new(wt.file(), wc.rank()))),
        };

        if turn == Color::Black {
            self.board.fullmoves += 1;
        }

        if role == Pawn || capture.is_some() {
            self.board.halfmoves = 0;
            self.history = Default::default();
        } else {
            self.board.halfmoves += 1;
            let entries = self.history[turn as usize].len();
            self.history[turn as usize].copy_within(..entries - 1, 1);
            self.history[turn as usize][0] = NonZeroU32::new(self.zobrist().cast());
        }

        self.board.turn = !self.board.turn;
        self.zobrist ^= ZobristNumbers::turn();

        if let Some(ep) = self.board.en_passant.take() {
            self.zobrist ^= ZobristNumbers::en_passant(ep.file());
        }

        if let Some((victim, target)) = capture {
            self.board.toggle(Piece::new(victim, !turn), target);
            self.zobrist ^= ZobristNumbers::psq(!turn, victim, target);
        }

        self.board.toggle(Piece::new(role, turn), wc);
        self.board.toggle(Piece::new(role, turn), wt);

        self.zobrist ^= ZobristNumbers::psq(turn, role, wc);
        self.zobrist ^= ZobristNumbers::psq(turn, role, wt);

        if let Some(promotion) = promotion {
            self.board.toggle(Piece::new(Pawn, turn), wt);
            self.board.toggle(Piece::new(promotion, turn), wt);
            self.zobrist ^= ZobristNumbers::psq(turn, Pawn, wt);
            self.zobrist ^= ZobristNumbers::psq(turn, promotion, wt);
        } else if role == Pawn && (wt - wc).abs() == 16 {
            self.board.en_passant = Some(Square::new(wc.file(), Rank::Third.perspective(turn)));
            self.zobrist ^= ZobristNumbers::en_passant(wc.file());
        } else if role == King && (wt - wc).abs() == 2 {
            let (wc, wt) = if wt > wc {
                (H1.perspective(turn), F1.perspective(turn))
            } else {
                (A1.perspective(turn), D1.perspective(turn))
            };

            self.board.toggle(Piece::new(Rook, turn), wc);
            self.board.toggle(Piece::new(Rook, turn), wt);
            self.zobrist ^= ZobristNumbers::psq(turn, Rook, wc);
            self.zobrist ^= ZobristNumbers::psq(turn, Rook, wt);
        }

        let disrupted = Castles::from(wc) | Castles::from(wt);
        if self.castles() & disrupted != Castles::none() {
            self.zobrist ^= ZobristNumbers::castling(self.castles());
            self.board.castles &= !disrupted;
            self.zobrist ^= ZobristNumbers::castling(self.castles());
        }

        let king = self.king(!turn);
        let ours = self.material(turn);
        let occupied = self.occupied();

        self.pinned = Bitboard::empty();
        self.checkers = match promotion.unwrap_or(role) {
            r @ Pawn | r @ Knight => Piece::new(r, !turn).attacks(king, occupied) & wt.into(),
            _ => Bitboard::empty(),
        };

        let queens = self.by_role(Queen);
        for role in [Bishop, Rook] {
            let slider = Piece::new(role, !turn);
            for wc in ours & slider.attacks(king, ours) & (queens | self.by_role(role)) {
                let blockers = occupied & Bitboard::segment(king, wc);
                match blockers.len() {
                    0 => self.checkers |= wc.bitboard(),
                    1 => self.pinned |= blockers,
                    _ => {}
                }
            }
        }

        (role, capture)
    }

    /// Play a [null-move].
    ///
    /// [null-move]: https://www.chessprogramming.org/Null_Move
    #[inline(always)]
    pub fn pass(&mut self) {
        debug_assert!(!self.is_check());

        let turn = self.turn();
        if turn == Color::Black {
            self.board.fullmoves += 1;
        }

        self.board.halfmoves += 1;
        let entries = self.history[turn as usize].len();
        self.history[turn as usize].copy_within(..entries - 1, 1);
        self.history[turn as usize][0] = NonZeroU32::new(self.zobrist.cast());

        self.board.turn = !self.board.turn;
        self.zobrist ^= ZobristNumbers::turn();
        if let Some(ep) = self.board.en_passant.take() {
            self.zobrist ^= ZobristNumbers::en_passant(ep.file());
        }

        let king = self.king(!turn);
        let ours = self.material(turn);
        let occupied = self.occupied();

        self.pinned = Bitboard::empty();
        let queens = self.by_role(Role::Queen);
        for role in [Role::Bishop, Role::Rook] {
            let slider = Piece::new(role, !turn);
            for wc in ours & slider.attacks(king, ours) & (queens | self.by_role(role)) {
                let blockers = occupied & Bitboard::segment(king, wc);
                if blockers.len() == 1 {
                    self.pinned |= blockers;
                }
            }
        }
    }

    /// Counts the total number of reachable positions to the given depth.
    pub fn perft(&self, depth: Depth) -> usize {
        match depth.get() {
            0 => 1,
            1 => self.moves().into_iter().map(|ms| ms.iter().len()).sum(),
            _ => self
                .moves()
                .unpack()
                .map(|m| {
                    let mut next = self.clone();
                    next.play(m);
                    next.perft(depth - 1)
                })
                .sum(),
        }
    }
}

impl Display for Position {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.board, f)
    }
}

/// The reason why parsing the FEN string failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error, From)]
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
        use {ParsePositionError::*, Role::*};

        let board: Board = s.parse()?;
        let king = board.king(board.turn).ok_or(IllegalPosition)?;
        let ours = board.material(board.turn);
        let theirs = board.material(!board.turn);
        let occupied = ours | theirs;

        let mut checkers = Bitboard::empty();
        for role in [Pawn, Knight] {
            let stepper = Piece::new(role, board.turn);
            checkers |= theirs & board.by_role(role) & stepper.attacks(king, occupied);
        }

        let mut pinned = Bitboard::empty();
        let queens = board.by_role(Queen);
        for role in [Bishop, Rook] {
            let slider = Piece::new(role, board.turn);
            for wc in theirs & slider.attacks(king, theirs) & (queens | board.by_role(role)) {
                let blockers = occupied & Bitboard::segment(king, wc);
                match blockers.len() {
                    0 => checkers |= wc.bitboard(),
                    1 => pinned |= blockers,
                    _ => {}
                }
            }
        }

        Ok(Position {
            checkers,
            pinned,
            zobrist: board.zobrist(),
            history: Default::default(),
            board,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cmp::Reverse, fmt::Debug, hash::DefaultHasher};
    use test_strategy::proptest;

    #[proptest]
    fn position_compares_by_board(a: Position, b: Position) {
        assert_eq!(a == b, a.board == b.board);
    }

    #[proptest]
    fn hash_is_consistent(a: Position, b: Position) {
        let mut hasher = DefaultHasher::default();
        a.hash(&mut hasher);
        let x = hasher.finish();

        let mut hasher = DefaultHasher::default();
        b.hash(&mut hasher);
        let y = hasher.finish();

        assert!(x != y || a == b);
    }

    #[proptest]
    fn occupied_returns_non_empty_squares(pos: Position) {
        for sq in pos.occupied() {
            assert_ne!(pos.piece_on(sq), None);
        }
    }

    #[proptest]
    fn material_is_either_pawn_or_piece(pos: Position, c: Color) {
        assert_eq!(pos.material(c), pos.pawns(c) ^ pos.pieces(c));
    }

    #[proptest]
    fn king_returns_square_occupied_by_a_king(pos: Position, c: Color) {
        assert_eq!(pos.piece_on(pos.king(c)), Some(Piece::new(Role::King, c)));
    }

    #[proptest]
    fn iter_returns_pieces_and_squares(pos: Position) {
        assert_eq!(Vec::from_iter(pos.iter()), Vec::from_iter(pos.board.iter()));
    }

    #[proptest]
    fn zobrist_hashes_the_board(pos: Position) {
        assert_eq!(pos.zobrist(), pos.board.zobrist());
    }

    #[proptest]
    fn checkmate_implies_outcome(pos: Position) {
        assert!(!pos.is_checkmate() || pos.outcome() == Some(Outcome::Checkmate(!pos.turn())));
    }

    #[proptest]
    fn stalemate_implies_outcome(pos: Position) {
        assert!(!pos.is_stalemate() || pos.outcome() == Some(Outcome::Stalemate));
    }

    #[proptest]
    fn checkmate_implies_check(pos: Position) {
        assert!(!pos.is_checkmate() || pos.is_check());
    }

    #[proptest]
    fn checkmate_and_stalemate_are_mutually_exclusive(pos: Position) {
        assert!(!(pos.is_checkmate() && pos.is_stalemate()));
    }

    #[proptest]
    fn check_and_stalemate_are_mutually_exclusive(pos: Position) {
        assert!(!(pos.is_check() && pos.is_stalemate()));
    }

    #[proptest]
    fn moves_returns_legal_moves_from_this_position(
        #[filter(#pos.outcome().is_none())] pos: Position,
    ) {
        for m in pos.moves().unpack() {
            pos.clone().play(m);
        }
    }

    #[proptest]
    fn exchanges_iterator_is_sorted_by_captor_of_least_value(
        #[filter(#pos.outcome().is_none())] pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
    ) {
        let sq = m.whither();
        let exchanges = pos.exchanges(m);
        let mut pos = pos.clone();
        pos.play(m);

        for (m, captor, victim) in exchanges {
            prop_assume!(pos.pinned().is_empty());
            prop_assume!(pos.checkers().is_empty());

            assert_eq!(
                Some((Some(captor), m.promotion())),
                pos.moves()
                    .unpack_if(|m| m.whither().contains(sq))
                    .map(|m| (pos.role_on(m.whence()), m.promotion()))
                    .min_by_key(|&(r, p)| (r, Reverse(p)))
            );

            assert!(matches!(pos.play(m), (c, Some((v, _))) if c == captor && v == victim));
        }
    }

    #[proptest]
    fn captures_reduce_material(
        #[filter(#pos.moves().unpack().any(|m| m.is_capture()))] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().unpack_if(|ms| ms.is_capture())))] m: Move,
    ) {
        let prev = pos.clone();
        pos.play(m);
        assert!(pos.material(pos.turn()).len() < prev.material(pos.turn()).len());
    }

    #[proptest]
    fn promotions_exchange_pawns(
        #[filter(#pos.moves().unpack().any(|m| m.is_promotion()))] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().unpack_if(|ms| ms.is_promotion())))] m: Move,
    ) {
        let prev = pos.clone();
        pos.play(m);

        assert!(pos.pawns(prev.turn()).len() < prev.pawns(prev.turn()).len());

        assert_eq!(
            pos.material(prev.turn()).len(),
            prev.material(prev.turn()).len()
        );
    }

    #[proptest]
    fn legal_move_updates_position(
        #[filter(#pos.outcome().is_none())] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().unpack()))] m: Move,
    ) {
        let prev = pos.clone();
        pos.play(m);

        assert_ne!(pos, prev);
        assert_ne!(pos.turn(), prev.turn());

        assert_eq!(pos.piece_on(m.whence()), None);
        assert_eq!(
            pos.piece_on(m.whither()),
            m.promotion()
                .map(|r| Piece::new(r, prev.turn()))
                .or_else(|| prev.piece_on(m.whence()))
        );

        assert_eq!(
            pos.occupied(),
            Role::iter().fold(Bitboard::empty(), |bb, r| bb | pos.by_role(r))
        );

        assert_eq!(
            pos.material(Color::White) & pos.material(Color::Black),
            Bitboard::empty()
        );

        for r in Role::iter() {
            for sq in Role::iter() {
                if r != sq {
                    assert_eq!(pos.by_role(r) & pos.by_role(sq), Bitboard::empty());
                }
            }
        }

        assert_eq!(
            pos.material(prev.turn()).len(),
            prev.material(prev.turn()).len()
        );

        assert_eq!(
            pos.material(pos.turn()).len(),
            prev.material(pos.turn()).len() - m.is_capture() as usize
        );

        if let Some(ep) = pos.en_passant() {
            assert_eq!(ep.rank(), Rank::Sixth.perspective(pos.turn()));
        }
    }

    #[proptest]
    #[should_panic]
    fn play_panics_if_move_illegal(
        mut pos: Position,
        #[filter(!#pos.moves().unpack().any(|m| #m == m))] m: Move,
    ) {
        pos.play(m);
    }

    #[proptest]
    fn pass_updates_position(#[filter(!#pos.is_check())] mut pos: Position) {
        let prev = pos.clone();
        pos.pass();
        assert_ne!(pos, prev);
    }

    #[proptest]
    fn pass_reverts_itself(#[filter(!#pos.is_check() )] mut pos: Position) {
        let prev = pos.clone();
        pos.pass();
        pos.pass();
        assert_eq!(Vec::from_iter(pos.iter()), Vec::from_iter(prev.iter()));
        assert_eq!(pos.checkers(), prev.checkers());
        assert_eq!(pos.pinned(), prev.pinned());
    }

    #[proptest]
    #[should_panic]
    fn pass_panics_if_in_check(#[filter(#pos.is_check())] mut pos: Position) {
        pos.pass();
    }

    #[proptest]
    fn threefold_repetition_implies_draw(#[filter(#pos.outcome().is_none() )] mut pos: Position) {
        let zobrist = NonZeroU32::new(pos.zobrist().cast());
        prop_assume!(zobrist.is_some());

        pos.history[pos.turn() as usize][..2].clone_from_slice(&[zobrist, zobrist]);
        assert!(pos.is_draw_by_repetition());
        assert_eq!(pos.outcome(), Some(Outcome::DrawByThreefoldRepetition));
    }

    #[proptest]
    fn parsing_printed_position_is_an_identity(pos: Position) {
        assert_eq!(pos.to_string().parse(), Ok(pos));
    }

    #[proptest]
    fn parsing_position_fails_for_invalid_board(#[filter(#s.parse::<Board>().is_err())] s: String) {
        assert_eq!(
            s.parse::<Position>().err(),
            s.parse::<Board>().err().map(ParsePositionError::InvalidFen)
        );
    }

    #[proptest]
    fn parsing_position_fails_for_illegal_board(#[filter(#b.king(#b.turn).is_none())] b: Board) {
        assert_eq!(
            b.to_string().parse::<Position>(),
            Err(ParsePositionError::IllegalPosition)
        );
    }

    #[rustfmt::skip]
    #[cfg(not(coverage))]
    const PERFT_SUITE: &[(&str, u8, usize)] = &[
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
    fn perft_counts_all_reachable_positions_up_to_ply(
        #[strategy(select(PERFT_SUITE))] entry: (&'static str, u8, usize),
    ) {
        let (fen, plies, count) = entry;
        let pos: Position = fen.parse()?;
        assert_eq!(pos.perft(plies.saturate()), count);
    }
}
