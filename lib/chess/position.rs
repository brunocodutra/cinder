use crate::util::{Assume, Integer};
use crate::{chess::*, search::Depth};
use arrayvec::{ArrayVec, CapacityError};
use derive_more::with_trait::{Debug, Display, Error, From};
use std::fmt::{self, Formatter};
use std::hash::{Hash, Hasher};
use std::{num::NonZeroU32, ops::Index, str::FromStr};

#[cfg(test)]
use proptest::{prelude::*, sample::*};

#[inline(always)]
fn collect_moves<const N: usize>(
    piece: Piece,
    wc: Square,
    wt: Bitboard,
    partition: Bitboard,
    buffer: &mut ArrayVec<MoveSet, N>,
) -> Result<(), CapacityError<MoveSet>> {
    let captures = wt & partition;
    let regulars = wt & !partition;

    if !captures.is_empty() {
        buffer.try_push(MoveSet::capture(piece, wc, captures))?;
    }

    if !regulars.is_empty() {
        buffer.try_push(MoveSet::regular(piece, wc, regulars))?;
    }

    Ok(())
}

struct Evasions;

impl Evasions {
    #[inline(always)]
    fn generate<const N: usize>(
        pos: &Position,
        buffer: &mut ArrayVec<MoveSet, N>,
    ) -> Result<(), CapacityError<MoveSet>> {
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

        for wc in candidates & pos.board.by_role(Role::Pawn) {
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

            collect_moves(piece, wc, moves, theirs | ep, buffer)?;
        }

        {
            let piece = Piece::new(Role::Knight, turn);
            for wc in candidates & pos.board.by_role(Role::Knight) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                collect_moves(piece, wc, moves, theirs, buffer)?;
            }
        }

        {
            let piece = Piece::new(Role::Bishop, turn);
            for wc in candidates & pos.board.by_role(Role::Bishop) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                collect_moves(piece, wc, moves, theirs, buffer)?;
            }
        }

        {
            let piece = Piece::new(Role::Rook, turn);
            for wc in candidates & pos.board.by_role(Role::Rook) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                collect_moves(piece, wc, moves, theirs, buffer)?;
            }
        }

        {
            let piece = Piece::new(Role::Queen, turn);
            for wc in candidates & pos.board.by_role(Role::Queen) {
                let moves = piece.moves(wc, ours, theirs) & (checks | pos.checkers());
                collect_moves(piece, wc, moves, theirs, buffer)?;
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

            collect_moves(piece, king, moves, theirs, buffer)?;
        }

        Ok(())
    }
}

struct Moves;

impl Moves {
    #[inline(always)]
    fn generate<const N: usize>(
        pos: &Position,
        buffer: &mut ArrayVec<MoveSet, N>,
    ) -> Result<(), CapacityError<MoveSet>> {
        let turn = pos.turn();
        let ours = pos.material(turn);
        let theirs = pos.material(!turn);
        let occupied = ours | theirs;
        let king = pos.king(turn);

        for wc in ours & pos.board.by_role(Role::Pawn) {
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

            collect_moves(piece, wc, moves, theirs | ep, buffer)?;
        }

        {
            let piece = Piece::new(Role::Knight, turn);
            for wc in ours & pos.board.by_role(Role::Knight) & !pos.pinned() {
                let moves = piece.moves(wc, ours, theirs);
                collect_moves(piece, wc, moves, theirs, buffer)?;
            }
        }

        {
            let piece = Piece::new(Role::Bishop, turn);
            for wc in ours & pos.board.by_role(Role::Bishop) {
                let mut moves = piece.moves(wc, ours, theirs);
                if pos.pinned().contains(wc) {
                    moves &= Bitboard::line(king, wc);
                }

                collect_moves(piece, wc, moves, theirs, buffer)?;
            }
        }

        {
            let piece = Piece::new(Role::Rook, turn);
            for wc in ours & pos.board.by_role(Role::Rook) {
                let mut moves = piece.moves(wc, ours, theirs);
                if pos.pinned().contains(wc) {
                    moves &= Bitboard::line(king, wc);
                }

                collect_moves(piece, wc, moves, theirs, buffer)?;
            }
        }

        {
            let piece = Piece::new(Role::Queen, turn);
            for wc in ours & pos.board.by_role(Role::Queen) {
                let mut moves = piece.moves(wc, ours, theirs);
                if pos.pinned().contains(wc) {
                    moves &= Bitboard::line(king, wc);
                }

                collect_moves(piece, wc, moves, theirs, buffer)?;
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

            collect_moves(piece, king, moves, theirs, buffer)?;
        }

        Ok(())
    }
}

/// The current position on the board board.
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
                        pos.play(selector.select(pos.moves().flatten()));
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
        self.board.by_color(side)
    }

    /// [`Square`]s occupied by pawns of a [`Color`].
    #[inline(always)]
    pub fn pawns(&self, side: Color) -> Bitboard {
        self.board.by_piece(Piece::new(Role::Pawn, side))
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

    /// An iterator over all pieces on the board.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (Piece, Square)> + '_ {
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
            let candidates = occupied & theirs & self.board.by_role(role);
            if Piece::new(role, !side).attacks(sq, occupied) & candidates != Bitboard::empty() {
                return true;
            }
        }

        let queens = self.board.by_role(Role::Queen);
        for role in [Role::Bishop, Role::Rook] {
            let candidates = occupied & theirs & (queens | self.board.by_role(role));
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
        self.is_check() && Evasions::generate(self, &mut ArrayVec::<_, 0>::new()).is_ok()
    }

    /// Whether this position is a [stalemate].
    ///
    /// [stalemate]: https://www.chessprogramming.org/Stalemate
    #[inline(always)]
    pub fn is_stalemate(&self) -> bool {
        !self.is_check() && Moves::generate(self, &mut ArrayVec::<_, 0>::new()).is_ok()
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
            3 => self.board.by_role(Bishop) | self.board.by_role(Knight) != Bitboard::empty(),
            4 => {
                let wb = self.board.by_piece(WhiteBishop);
                let bb = self.board.by_piece(BlackBishop);

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

    /// An iterator over the legal moves that can be played in this position.
    #[inline(always)]
    pub fn moves(&self) -> impl Iterator<Item = MoveSet> {
        let mut moves = ArrayVec::<_, 32>::new();

        if self.is_check() {
            Evasions::generate(self, &mut moves).assume()
        } else {
            Moves::generate(self, &mut moves).assume()
        }

        moves.into_iter()
    }

    /// The sequence of captures om a square starting from a move ordered by least valued captor.
    ///
    /// Pins and checks are ignored.
    #[inline(always)]
    pub fn exchanges(&self, m: Move) -> impl Iterator<Item = (Move, Role, Role)> {
        use {Color::*, Piece::*, Role::*};

        gen move {
            let sq = m.whither();
            let queens = self.board.by_role(Queen);
            let rooks = self.board.by_role(Rook);
            let bishops = self.board.by_role(Bishop);

            let mut turn = self.turn();
            let mut attackers = Bitboard::empty();
            let mut occupied = self.occupied().without(m.whence()).without(sq);
            let mut victim = match m.promotion() {
                None => self.board.role_on(m.whence()).assume(),
                Some(r) => r,
            };

            for piece in [WhitePawn, BlackPawn] {
                attackers |= self.board.by_piece(piece) & piece.flip().attacks(sq, occupied);
            }

            for role in [Knight, King] {
                let candidates = self.board.by_role(role);
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
                    let bb = candidates & self.board.by_role(role);
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
        debug_assert!(self.moves().flatten().any(|n| m == n));

        use {Role::*, Square::*};

        let turn = self.turn();
        let promotion = m.promotion();
        let (wc, wt) = (m.whence(), m.whither());
        let role = self.board.role_on(wc).assume();
        let capture = match self.board.role_on(wt) {
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

        let queens = self.board.by_role(Queen);
        for role in [Bishop, Rook] {
            let slider = Piece::new(role, !turn);
            for wc in ours & slider.attacks(king, ours) & (queens | self.board.by_role(role)) {
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
        let queens = self.board.by_role(Role::Queen);
        for role in [Role::Bishop, Role::Rook] {
            let slider = Piece::new(role, !turn);
            for wc in ours & slider.attacks(king, ours) & (queens | self.board.by_role(role)) {
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
            1 => self.moves().map(|ms| ms.iter().len()).sum(),
            _ => self
                .moves()
                .flatten()
                .map(|m| {
                    let mut next = self.clone();
                    next.play(m);
                    next.perft(depth - 1)
                })
                .sum(),
        }
    }
}

/// Retrieves the [`Piece`] at a given [`Square`], if any.
impl Index<Square> for Position {
    type Output = Option<Piece>;

    #[inline(always)]
    fn index(&self, sq: Square) -> &Self::Output {
        match self.board.piece_on(sq) {
            Some(Piece::WhitePawn) => &Some(Piece::WhitePawn),
            Some(Piece::WhiteKnight) => &Some(Piece::WhiteKnight),
            Some(Piece::WhiteBishop) => &Some(Piece::WhiteBishop),
            Some(Piece::WhiteRook) => &Some(Piece::WhiteRook),
            Some(Piece::WhiteQueen) => &Some(Piece::WhiteQueen),
            Some(Piece::WhiteKing) => &Some(Piece::WhiteKing),
            Some(Piece::BlackPawn) => &Some(Piece::BlackPawn),
            Some(Piece::BlackKnight) => &Some(Piece::BlackKnight),
            Some(Piece::BlackBishop) => &Some(Piece::BlackBishop),
            Some(Piece::BlackRook) => &Some(Piece::BlackRook),
            Some(Piece::BlackQueen) => &Some(Piece::BlackQueen),
            Some(Piece::BlackKing) => &Some(Piece::BlackKing),
            None => &None,
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
        let ours = board.by_color(board.turn);
        let theirs = board.by_color(!board.turn);
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
            assert_ne!(pos[sq], None);
        }
    }

    #[proptest]
    fn material_is_either_pawn_or_piece(pos: Position, c: Color) {
        assert_eq!(pos.material(c), pos.pawns(c) ^ pos.pieces(c));
    }

    #[proptest]
    fn king_returns_square_occupied_by_a_king(pos: Position, c: Color) {
        assert_eq!(pos[pos.king(c)], Some(Piece::new(Role::King, c)));
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
        for m in pos.moves().flatten() {
            pos.clone().play(m);
        }
    }

    #[proptest]
    fn exchanges_iterator_is_sorted_by_captor_of_least_value(
        #[filter(#pos.outcome().is_none())] pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
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
                    .filter(|m| m.whither().contains(sq))
                    .flatten()
                    .map(|m| (pos.board.role_on(m.whence()), m.promotion()))
                    .min_by_key(|&(r, p)| (r, Reverse(p)))
            );

            assert!(matches!(pos.play(m), (c, Some((v, _))) if c == captor && v == victim));
        }
    }

    #[proptest]
    fn captures_reduce_material(
        #[filter(#pos.moves().any(|ms| ms.is_capture()))] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().filter(|ms| ms.is_capture()).flatten()))] m: Move,
    ) {
        let prev = pos.clone();
        pos.play(m);
        assert!(pos.material(pos.turn()).len() < prev.material(pos.turn()).len());
    }

    #[proptest]
    fn promotions_exchange_pawns(
        #[filter(#pos.moves().any(|ms| ms.is_promotion()))] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().filter(|ms| ms.is_promotion()).flatten()))]
        m: Move,
    ) {
        let prev = pos.clone();
        pos.play(m);
        let pawn = Piece::new(Role::Pawn, prev.turn());
        assert!(pos.board.by_piece(pawn).len() < prev.board.by_piece(pawn).len());

        assert_eq!(
            pos.material(prev.turn()).len(),
            prev.material(prev.turn()).len()
        );
    }

    #[proptest]
    fn legal_move_updates_position(
        #[filter(#pos.outcome().is_none())] mut pos: Position,
        #[map(|s: Selector| s.select(#pos.moves().flatten()))] m: Move,
    ) {
        let prev = pos.clone();
        pos.play(m);

        assert_ne!(pos, prev);
        assert_ne!(pos.turn(), prev.turn());

        assert_eq!(pos[m.whence()], None);
        assert_eq!(
            pos[m.whither()],
            m.promotion()
                .map(|r| Piece::new(r, prev.turn()))
                .or_else(|| prev[m.whence()])
        );

        assert_eq!(
            pos.occupied(),
            Role::iter().fold(Bitboard::empty(), |bb, r| bb | pos.board.by_role(r))
        );

        assert_eq!(
            pos.material(Color::White) & pos.material(Color::Black),
            Bitboard::empty()
        );

        for r in Role::iter() {
            for sq in Role::iter() {
                if r != sq {
                    assert_eq!(
                        pos.board.by_role(r) & pos.board.by_role(sq),
                        Bitboard::empty()
                    );
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
        #[filter(!#pos.moves().flatten().any(|m| #m == m))] m: Move,
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

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    fn perft_1() {
        // https://www.chessprogramming.org/Perft_Results#Initial_Position
        assert_eq!(Position::default().perft(Depth::new(5)), 4865609);
    }

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    fn perft_2() {
        // https://www.chessprogramming.org/Perft_Results#Position_2
        let pos = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1".parse()?;
        assert_eq!(Position::perft(&pos, Depth::new(5)), 193690690);
    }

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    fn perft_3() {
        // https://www.chessprogramming.org/Perft_Results#Position_3
        let pos = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1".parse()?;
        assert_eq!(Position::perft(&pos, Depth::new(5)), 674624);
    }

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    fn perft_4() {
        // https://www.chessprogramming.org/Perft_Results#Position_4
        let pos = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1".parse()?;
        assert_eq!(Position::perft(&pos, Depth::new(5)), 15833292);
    }

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    fn perft_5() {
        // https://www.chessprogramming.org/Perft_Results#Position_5
        let pos = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8".parse()?;
        assert_eq!(Position::perft(&pos, Depth::new(5)), 89941194);
    }

    #[cfg(not(coverage))]
    #[proptest(cases = 1)]
    fn perft_6() {
        // https://www.chessprogramming.org/Perft_Results#Position_6
        let pos =
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10".parse()?;
        assert_eq!(Position::perft(&pos, Depth::new(5)), 164075551);
    }
}
