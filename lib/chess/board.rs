use crate::chess::*;
use crate::util::{Assume, Integer};
use derive_more::with_trait::{Debug, Display, Error};
use std::fmt::{self, Formatter, Write};
use std::io::Write as _;
use std::str::{self, FromStr};

#[cfg(test)]
use proptest::strategy::LazyJust;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Zobrists {
    pub hash: Zobrist,
    pub pawns: Zobrist,
    pub minor: Zobrist,
    pub major: Zobrist,
    pub white: Zobrist,
    pub black: Zobrist,
}

impl Zobrists {
    #[inline(always)]
    pub fn toggle(&mut self, p: Piece, sq: Square) {
        self.hash ^= ZobristNumbers::psq(p, sq);

        if p.role() == Role::Pawn {
            self.pawns ^= ZobristNumbers::psq(p, sq);
        } else {
            match p.color() {
                Color::White => self.white ^= ZobristNumbers::psq(p, sq),
                Color::Black => self.black ^= ZobristNumbers::psq(p, sq),
            }

            if matches!(p.role(), Role::Knight | Role::Bishop) {
                self.minor ^= ZobristNumbers::psq(p, sq);
            } else if matches!(p.role(), Role::Rook | Role::Queen) {
                self.major ^= ZobristNumbers::psq(p, sq);
            } else {
                self.minor ^= ZobristNumbers::psq(p, sq);
                self.major ^= ZobristNumbers::psq(p, sq);
            }
        }
    }
}

/// The chess board.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Board({self})")]
pub struct Board {
    #[cfg_attr(test, strategy(LazyJust::new(move || {
        let mut roles = [Bitboard::empty(); 6];
        for (i, o) in #pieces.iter().enumerate() {
            if let Some(p) = o {
                roles[p.role() as usize] |= <Square as Integer>::new(i as _).bitboard();
            }
        }

        roles
    })))]
    roles: [Bitboard; 6],
    #[cfg_attr(test, strategy(LazyJust::new(move || {
        let mut colors = [Bitboard::empty(); 2];
        for (i, o) in #pieces.iter().enumerate() {
            if let Some(p) = o {
                colors[p.color() as usize] |= <Square as Integer>::new(i as _).bitboard();
            }
        }

        colors
    })))]
    colors: [Bitboard; 2],
    pieces: [Option<Piece>; 64],
    pub turn: Color,
    pub castles: Castles,
    pub en_passant: Option<Square>,
    pub halfmoves: u8,
    pub fullmoves: u32,
}

impl Default for Board {
    #[inline(always)]
    fn default() -> Self {
        use Piece::*;

        #[rustfmt::skip]
        let pieces = [
            Some(WhiteRook), Some(WhiteKnight), Some(WhiteBishop), Some(WhiteQueen), Some(WhiteKing), Some(WhiteBishop), Some(WhiteKnight), Some(WhiteRook),
            Some(WhitePawn), Some(WhitePawn),   Some(WhitePawn),   Some(WhitePawn),  Some(WhitePawn), Some(WhitePawn),   Some(WhitePawn),   Some(WhitePawn),
            None,            None,              None,              None,             None,            None,              None,              None,
            None,            None,              None,              None,             None,            None,              None,              None,
            None,            None,              None,              None,             None,            None,              None,              None,
            None,            None,              None,              None,             None,            None,              None,              None,
            Some(BlackPawn), Some(BlackPawn),   Some(BlackPawn),   Some(BlackPawn),  Some(BlackPawn), Some(BlackPawn),   Some(BlackPawn),   Some(BlackPawn),
            Some(BlackRook), Some(BlackKnight), Some(BlackBishop), Some(BlackQueen), Some(BlackKing), Some(BlackBishop), Some(BlackKnight), Some(BlackRook),
        ];

        Self {
            roles: [
                Bitboard::new(0x00FF00000000FF00),
                Bitboard::new(0x4200000000000042),
                Bitboard::new(0x2400000000000024),
                Bitboard::new(0x8100000000000081),
                Bitboard::new(0x0800000000000008),
                Bitboard::new(0x1000000000000010),
            ],
            colors: [
                Bitboard::new(0x000000000000FFFF),
                Bitboard::new(0xFFFF000000000000),
            ],
            pieces,
            turn: Color::White,
            castles: Castles::all(),
            en_passant: None,
            halfmoves: 0,
            fullmoves: 1,
        }
    }
}

impl Board {
    /// [`Square`]s occupied by [`Piece`]s of a [`Color`].
    #[inline(always)]
    pub fn material(&self, c: Color) -> Bitboard {
        self.colors[c as usize]
    }

    /// [`Square`]s occupied by [`Piece`]s of a [`Role`].
    #[inline(always)]
    pub fn by_role(&self, r: Role) -> Bitboard {
        self.roles[r as usize]
    }

    /// [`Square`]s occupied by a [`Piece`].
    #[inline(always)]
    pub fn by_piece(&self, p: Piece) -> Bitboard {
        self.material(p.color()) & self.by_role(p.role())
    }

    /// Squares occupied by pinned [`Piece`]s of a [`Color`].
    #[inline(always)]
    pub fn pinned(&self, c: Color, mask: Bitboard) -> Bitboard {
        let ours = mask & self.material(c);
        let theirs = mask & self.material(!c);
        let occ = ours ^ theirs;

        let king = self.king(c).assume();
        let queens = self.by_role(Role::Queen);

        let mut pinned = Bitboard::empty();
        for role in [Role::Bishop, Role::Rook] {
            let slider = Piece::new(role, c);
            for wc in theirs & slider.attacks(king, theirs) & (queens | self.by_role(role)) {
                let blockers = occ & Bitboard::segment(king, wc);
                if blockers.len() == 1 {
                    pinned |= blockers & ours;
                }
            }
        }

        pinned
    }

    /// Squares occupied by [`Piece`]s checking the king of a [`Color`].
    #[inline(always)]
    pub fn checkers(&self, c: Color) -> Bitboard {
        let ours = self.material(c);
        let theirs = self.material(!c);
        let occ = ours ^ theirs;

        let king = self.king(c).assume();
        let queens = self.by_role(Role::Queen);

        let mut checkers = Bitboard::empty();
        for role in [Role::Pawn, Role::Knight] {
            let stepper = Piece::new(role, c);
            checkers |= theirs & self.by_role(role) & stepper.attacks(king, occ);
        }

        for role in [Role::Bishop, Role::Rook] {
            let slider = Piece::new(role, c);
            for wc in theirs & slider.attacks(king, theirs) & (queens | self.by_role(role)) {
                let blockers = occ & Bitboard::segment(king, wc);
                if blockers.is_empty() {
                    checkers |= wc.bitboard()
                }
            }
        }

        checkers
    }

    /// Squares occupied by [`Square`]s threatened by [`Piece`]s of a [`Color`].
    #[inline(always)]
    pub fn threats(&self, c: Color) -> Bitboard {
        let ours = self.material(!c);
        let theirs = self.material(c);
        let occ = ours ^ theirs;

        let king = self.king(!c).assume();
        let pawns = theirs & self.by_role(Role::Pawn);
        let mut threats = Bitboard::empty();

        threats |= match c {
            Color::White => (pawns << 7 & !File::H.bitboard()) | (pawns << 9 & !File::A.bitboard()),
            Color::Black => (pawns >> 7 & !File::A.bitboard()) | (pawns >> 9 & !File::H.bitboard()),
        };

        let blockers = occ.without(king);
        for role in [Role::Knight, Role::King] {
            for wc in theirs & self.by_role(role) {
                threats |= Piece::new(role, c).attacks(wc, blockers);
            }
        }

        for role in [Role::Bishop, Role::Rook] {
            for wc in theirs & (self.by_role(role) | self.by_role(Role::Queen)) {
                threats |= Piece::new(role, c).attacks(wc, blockers);
            }
        }

        threats
    }

    /// [`Square`] occupied by a the king of a [`Color`].
    #[inline(always)]
    pub fn king(&self, side: Color) -> Option<Square> {
        let piece = Piece::new(Role::King, side);
        self.by_piece(piece).into_iter().next()
    }

    /// The [`Color`] of the piece on the given [`Square`], if any.
    #[inline(always)]
    pub fn color_on(&self, sq: Square) -> Option<Color> {
        self.piece_on(sq).map(|p| p.color())
    }

    /// The [`Role`] of the piece on the given [`Square`], if any.
    #[inline(always)]
    pub fn role_on(&self, sq: Square) -> Option<Role> {
        self.piece_on(sq).map(|p| p.role())
    }

    /// The [`Piece`] on the given [`Square`], if any.
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.pieces[sq as usize]
    }

    /// An iterator over all pieces on the board.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (Piece, Square)> + '_ {
        Piece::iter().flat_map(|p| self.by_piece(p).into_iter().map(move |sq| (p, sq)))
    }

    /// Computes the [zobrist hashes](`Zobrists`).
    #[inline(always)]
    pub fn zobrists(&self) -> Zobrists {
        let mut zobrists = Zobrists {
            hash: ZobristNumbers::castling(self.castles),
            ..Default::default()
        };

        if self.turn == Color::Black {
            zobrists.hash ^= ZobristNumbers::turn();
        }

        if let Some(ep) = self.en_passant {
            zobrists.hash ^= ZobristNumbers::en_passant(ep.file());
        }

        for (p, sq) in self.iter() {
            zobrists.toggle(p, sq);
        }

        zobrists
    }

    /// Toggles a piece on a square.
    #[inline(always)]
    pub fn toggle(&mut self, p: Piece, sq: Square) {
        debug_assert!(self.piece_on(sq).is_none_or(|q| p == q));
        self.pieces[sq as usize] = self.pieces[sq as usize].xor(Some(p));
        self.colors[p.color() as usize] ^= sq.bitboard();
        self.roles[p.role() as usize] ^= sq.bitboard();
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut skip = 0;
        for sq in Square::iter().map(|sq| sq.flip()) {
            let mut buffer = [b'\0'; 2];

            if sq.file() == File::H {
                buffer[0] = if sq.rank() == Rank::First { b' ' } else { b'/' };
            }

            match self.piece_on(sq) {
                None => skip += 1,
                Some(p) => {
                    buffer[1] = buffer[0];
                    write!(&mut buffer[..1], "{p}").assume()
                }
            }

            if skip > 0 && buffer != [b'\0'; 2] {
                write!(f, "{skip}")?;
                skip = 0;
            }

            for b in buffer.into_iter().take_while(|&b| b != b'\0') {
                f.write_char(b.into())?;
            }
        }

        match self.turn {
            Color::White => f.write_str("w ")?,
            Color::Black => f.write_str("b ")?,
        }

        if self.castles != Castles::none() {
            write!(f, "{} ", self.castles)?;
        } else {
            f.write_str("- ")?;
        }

        if let Some(ep) = self.en_passant {
            write!(f, "{ep} ")?;
        } else {
            f.write_str("- ")?;
        }

        write!(f, "{} {}", self.halfmoves, self.fullmoves)?;

        Ok(())
    }
}

/// The reason why parsing the FEN string failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
pub enum ParseFenError {
    #[display("failed to parse piece placement")]
    InvalidPlacement,
    #[display("failed to parse side to move")]
    InvalidSideToMove,
    #[display("failed to parse castling rights")]
    InvalidCastlingRights,
    #[display("failed to parse en passant square")]
    InvalidEnPassantSquare,
    #[display("failed to parse halfmove clock")]
    InvalidHalfmoveClock,
    #[display("failed to parse fullmove number")]
    InvalidFullmoveNumber,
    #[display("unspecified syntax error")]
    InvalidSyntax,
}

impl FromStr for Board {
    type Err = ParseFenError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = &mut s.split_ascii_whitespace();

        let Some(board) = tokens.next() else {
            return Err(ParseFenError::InvalidPlacement);
        };

        let mut pieces: [_; 64] = [None; 64];
        let mut roles: [_; 6] = Default::default();
        let mut colors: [_; 2] = Default::default();
        for (rank, segment) in board.split('/').rev().enumerate() {
            let mut file = 0;
            for c in segment.chars() {
                let mut buffer = [0; 4];

                if file >= 8 || rank >= 8 {
                    return Err(ParseFenError::InvalidPlacement);
                } else if let Some(skip) = c.to_digit(10) {
                    file += skip;
                } else if let Ok(p) = Piece::from_str(c.encode_utf8(&mut buffer)) {
                    let sq = Square::new(File::new(file as _), Rank::new(rank as _));
                    colors[p.color() as usize] ^= sq.bitboard();
                    roles[p.role() as usize] ^= sq.bitboard();
                    pieces[sq as usize] = Some(p);
                    file += 1;
                } else {
                    return Err(ParseFenError::InvalidPlacement);
                }
            }
        }

        let turn = match tokens.next() {
            Some("w") => Color::White,
            Some("b") => Color::Black,
            _ => return Err(ParseFenError::InvalidSideToMove),
        };

        let castles = match tokens.next() {
            None => return Err(ParseFenError::InvalidCastlingRights),
            Some("-") => Castles::none(),
            Some(s) => match s.parse() {
                Err(_) => return Err(ParseFenError::InvalidCastlingRights),
                Ok(castles) => castles,
            },
        };

        let en_passant = match tokens.next() {
            None => return Err(ParseFenError::InvalidEnPassantSquare),
            Some("-") => None,
            Some(ep) => match ep.parse() {
                Err(_) => return Err(ParseFenError::InvalidEnPassantSquare),
                Ok(sq) => Some(sq),
            },
        };

        let Some(Ok(halfmoves)) = tokens.next().map(u8::from_str) else {
            return Err(ParseFenError::InvalidHalfmoveClock);
        };

        let Some(Ok(fullmoves)) = tokens.next().map(u32::from_str) else {
            return Err(ParseFenError::InvalidHalfmoveClock);
        };

        if tokens.next().is_some() {
            return Err(ParseFenError::InvalidSyntax);
        }

        Ok(Board {
            roles,
            colors,
            pieces,
            turn,
            castles,
            en_passant,
            halfmoves,
            fullmoves,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn iter_returns_pieces_and_squares(b: Board) {
        for (p, sq) in b.iter() {
            assert_eq!(b.piece_on(sq), Some(p));
        }
    }

    #[proptest]
    fn by_color_returns_squares_occupied_by_pieces_of_a_color(b: Board, c: Color) {
        for sq in b.material(c) {
            assert_eq!(b.piece_on(sq).map(|p| p.color()), Some(c));
        }
    }

    #[proptest]
    fn by_color_returns_squares_occupied_by_pieces_of_a_role(b: Board, r: Role) {
        for sq in b.by_role(r) {
            assert_eq!(b.piece_on(sq).map(|p| p.role()), Some(r));
        }
    }

    #[proptest]
    fn by_piece_returns_squares_occupied_by_a_piece(b: Board, p: Piece) {
        for sq in b.by_piece(p) {
            assert_eq!(b.piece_on(sq), Some(p));
        }
    }

    #[proptest]
    fn king_returns_square_occupied_by_a_king(b: Board, c: Color) {
        if let Some(sq) = b.king(c) {
            assert_eq!(b.piece_on(sq), Some(Piece::new(Role::King, c)));
        }
    }

    #[proptest]
    fn piece_on_returns_piece_on_the_given_square(b: Board, sq: Square) {
        assert_eq!(
            b.piece_on(sq),
            Option::zip(b.color_on(sq), b.role_on(sq)).map(|(c, r)| Piece::new(r, c))
        );
    }

    #[proptest]
    fn toggle_removes_piece_from_square(
        mut b: Board,
        #[filter(#b.piece_on(#sq).is_some())] sq: Square,
    ) {
        let p = b.piece_on(sq).unwrap();
        b.toggle(p, sq);
        assert_eq!(b.piece_on(sq), None);
    }

    #[proptest]
    fn toggle_places_piece_on_square(
        mut b: Board,
        #[filter(#b.piece_on(#sq).is_none())] sq: Square,
        p: Piece,
    ) {
        b.toggle(p, sq);
        assert_eq!(b.piece_on(sq), Some(p));
    }

    #[proptest]
    #[should_panic]
    fn toggle_panics_if_square_occupied_by_other_piece(
        mut b: Board,
        #[filter(#b.piece_on(#sq).is_some())] sq: Square,
        #[filter(Some(#p) != #b.piece_on(#sq))] p: Piece,
    ) {
        b.toggle(p, sq);
    }

    #[proptest]
    fn parsing_printed_board_is_an_identity(b: Board) {
        assert_eq!(b.to_string().parse(), Ok(b));
    }

    #[proptest]
    fn parsing_board_fails_for_invalid_fen(
        b: Board,
        #[strategy(..=#b.to_string().len())] n: usize,
        #[strategy("[^[:ascii:]]+")] r: String,
    ) {
        let s = b.to_string();
        assert_eq!([&s[..n], &r, &s[n..]].concat().parse().ok(), None::<Board>);
    }
}
