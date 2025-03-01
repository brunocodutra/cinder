use crate::chess::*;
use crate::util::{Assume, Integer};
use derive_more::with_trait::{Debug, Display, Error};
use std::fmt::{self, Formatter, Write};
use std::io::Write as _;
use std::str::{self, FromStr};

/// The chess board.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Board({self})")]
pub struct Board {
    #[cfg_attr(test, map(|mut bbs: [Bitboard; 6]| {
        let mut occupied = bbs[0];
        for bb in &mut bbs[1..] {
            *bb &= !occupied;
            occupied |= *bb;
        }

        bbs
    }))]
    roles: [Bitboard; 6],
    #[cfg_attr(test, map(|bb: Bitboard| {
        let occupied = #roles.iter().fold(Bitboard::empty(), |r, &b| r | b);
        [occupied & bb, occupied & !bb]
    }))]
    colors: [Bitboard; 2],
    pub turn: Color,
    pub castles: Castles,
    pub en_passant: Option<Square>,
    pub halfmoves: u8,
    pub fullmoves: u32,
}

impl Default for Board {
    #[inline(always)]
    fn default() -> Self {
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

            turn: Color::White,
            castles: Castles::all(),
            en_passant: None,
            halfmoves: 0,
            fullmoves: 1,
        }
    }
}

impl Board {
    /// [`Square`]s occupied by a [`Color`].
    #[inline(always)]
    pub fn by_color(&self, c: Color) -> Bitboard {
        self.colors[c as usize]
    }

    /// [`Square`]s occupied by a [`Role`].
    #[inline(always)]
    pub fn by_role(&self, r: Role) -> Bitboard {
        self.roles[r as usize]
    }

    /// [`Square`]s occupied by a [`Piece`].
    #[inline(always)]
    pub fn by_piece(&self, p: Piece) -> Bitboard {
        self.by_color(p.color()) & self.by_role(p.role())
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
        Color::iter().find(|&c| self.by_color(c).contains(sq))
    }

    /// The [`Role`] of the piece on the given [`Square`], if any.
    #[inline(always)]
    pub fn role_on(&self, sq: Square) -> Option<Role> {
        Role::iter().find(|&r| self.by_role(r).contains(sq))
    }

    /// The [`Piece`] on the given [`Square`], if any.
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        Option::zip(self.role_on(sq), self.color_on(sq)).map(|(r, c)| Piece::new(r, c))
    }

    /// An iterator over all pieces on the board.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (Piece, Square)> + '_ {
        Piece::iter().flat_map(|p| self.by_piece(p).into_iter().map(move |sq| (p, sq)))
    }

    /// Computes the [zobrist hash].
    ///
    /// [zobrist hash]: https://www.chessprogramming.org/Zobrist_Hashing
    #[inline(always)]
    pub fn zobrist(&self) -> Zobrist {
        let mut zobrist = ZobristNumbers::castling(self.castles);

        for (p, sq) in self.iter() {
            zobrist ^= ZobristNumbers::psq(p.color(), p.role(), sq);
        }

        if self.turn == Color::Black {
            zobrist ^= ZobristNumbers::turn();
        }

        if let Some(ep) = self.en_passant {
            zobrist ^= ZobristNumbers::en_passant(ep.file());
        }

        zobrist
    }

    /// Toggles a piece on a square.
    #[inline(always)]
    pub fn toggle(&mut self, p: Piece, sq: Square) {
        debug_assert!(self.piece_on(sq).is_none_or(|q| p == q));
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
        for sq in b.by_color(c) {
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
