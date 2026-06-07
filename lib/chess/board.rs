use crate::util::{Assume, Int, Num};
use crate::{chess::*, simd::*};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, Display, Error};
use std::fmt::{self, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::str::{self, FromStr};
use std::{io::Write as _, ops::BitAnd};

#[cfg(test)]
use proptest::prelude::*;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Zeroable)]
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
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(board: &Board) -> Self {
        let mut zobrists = Zobrists {
            hash: ZobristNumbers::castling(board.castles),
            ..Default::default()
        };

        if board.turn == Color::Black {
            zobrists.hash ^= ZobristNumbers::turn();
        }

        if let Some(ep) = board.en_passant {
            zobrists.hash ^= ZobristNumbers::en_passant(ep.file());
        }

        for p in Piece::iter() {
            for sq in Bitboard::from(board.by_piece(p)) {
                zobrists.xor(sq, p);
            }
        }

        zobrists
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn xor(&mut self, sq: Square, p: Piece) {
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
#[derive(Debug, Clone, Copy, Eq, Deref)]
#[debug("Board({self})")]
pub struct Board {
    pub turn: Color,
    pub castles: Castles,
    pub en_passant: Option<Square>,
    pub halfmoves: u8,
    pub fullmoves: u32,

    #[deref]
    placement: Placement,
    squares: SquareByIdx,
    roles: RoleByIdx,
}

#[cfg(test)]
impl Arbitrary for Board {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        any::<Position>().prop_map_into().boxed()
    }
}

impl Default for Board {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn default() -> Self {
        use Piece::*;

        #[rustfmt::skip]
        let placement = Placement::new([
            Place::new(WhiteRook, Idx::new(5)), Place::new(WhiteKnight, Idx::new(3)), Place::new(WhiteBishop, Idx::new(1)), Place::new(WhiteQueen, Idx::new(7)), Place::new(WhiteKing, Idx::new(0)),  Place::new(WhiteBishop, Idx::new(2)), Place::new(WhiteKnight, Idx::new(4)), Place::new(WhiteRook, Idx::new(6)),
            Place::new(WhitePawn, Idx::new(8)), Place::new(WhitePawn, Idx::new(9)),   Place::new(WhitePawn, Idx::new(10)),  Place::new(WhitePawn, Idx::new(11)), Place::new(WhitePawn, Idx::new(12)), Place::new(WhitePawn, Idx::new(13)),  Place::new(WhitePawn, Idx::new(14)),  Place::new(WhitePawn, Idx::new(15)),
            zeroed(),                             zeroed(),                               zeroed(),                               zeroed(),                              zeroed(),                              zeroed(),                               zeroed(),                               zeroed(),
            zeroed(),                             zeroed(),                               zeroed(),                               zeroed(),                              zeroed(),                              zeroed(),                               zeroed(),                               zeroed(),
            zeroed(),                             zeroed(),                               zeroed(),                               zeroed(),                              zeroed(),                              zeroed(),                               zeroed(),                               zeroed(),
            zeroed(),                             zeroed(),                               zeroed(),                               zeroed(),                              zeroed(),                              zeroed(),                               zeroed(),                               zeroed(),
            Place::new(BlackPawn, Idx::new(8)), Place::new(BlackPawn, Idx::new(9)),   Place::new(BlackPawn, Idx::new(10)),  Place::new(BlackPawn, Idx::new(11)), Place::new(BlackPawn, Idx::new(12)), Place::new(BlackPawn, Idx::new(13)),  Place::new(BlackPawn, Idx::new(14)),  Place::new(BlackPawn, Idx::new(15)),
            Place::new(BlackRook, Idx::new(5)), Place::new(BlackKnight, Idx::new(3)), Place::new(BlackBishop, Idx::new(1)), Place::new(BlackQueen, Idx::new(7)), Place::new(BlackKing, Idx::new(0)),  Place::new(BlackBishop, Idx::new(2)), Place::new(BlackKnight, Idx::new(4)), Place::new(BlackRook, Idx::new(6)),
        ]);

        let squares = SquareByIdx::new(&placement);
        let roles = RoleByIdx::new(&placement);

        Self {
            turn: Color::White,
            castles: Castles::all(),
            en_passant: None,
            halfmoves: 0,
            fullmoves: 1,
            placement,
            squares,
            roles,
        }
    }
}

impl PartialEq for Board {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.turn == other.turn
            && self.castles == other.castles
            && self.en_passant == other.en_passant
            && self.halfmoves == other.halfmoves
            && self.fullmoves == other.fullmoves
            && self.placement.pieces() == other.placement.pieces()
    }
}

impl Hash for Board {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.zobrists().hash.hash(state);
    }
}

impl Board {
    /// Game [`Phase`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn phase(&self) -> Phase {
        Phase::new((self.occupied().count() - 1) as u8 / 4)
    }

    /// The [`Placement`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Squares by piece [`Idx`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn squares(&self) -> &SquareByIdx {
        &self.squares
    }

    /// Roles by piece [`Idx`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn roles(&self) -> &RoleByIdx {
        &self.roles
    }

    /// [`Square`]s occupied.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn occupied(&self) -> M8x64 {
        self.placement.occupied()
    }

    /// [`Square`]s vacant.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn vacant(&self) -> M8x64 {
        self.placement.vacant()
    }

    /// [`Square`]s occupied by [`Piece`]s of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_color(&self, c: Color) -> M8x64 {
        self.placement.by_color(c)
    }

    /// [`Square`]s occupied by [`Piece`]s of a [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_role(&self, r: Role) -> M8x64 {
        self.placement.by_role(r)
    }

    /// [`Square`]s occupied by a [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_piece(&self, p: Piece) -> M8x64 {
        self.placement.by_piece(p)
    }

    /// [`Square`] occupied by a the king of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn king(&self, side: Color) -> Option<Square> {
        self.squares[side][Idx::KING]
    }

    /// Squares giving direct check to the king of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn direct_checks(&self) -> [Bitboard; 4] {
        let them = !self.turn;
        let rays = self.king(them).assume().rays();
        let valid = rays.inv().valid();
        let furled = self.furl(rays);
        let mask = furled.visible();

        use Role::*;
        let pawn_attacks = mask & furled.attacks(Piece::new(Pawn, them));
        let knight_attacks = mask & furled.attacks(Piece::new(Knight, them));
        let bishop_attacks = mask & furled.attacks(Piece::new(Bishop, them));
        let rook_attacks = mask & furled.attacks(Piece::new(Rook, them));

        let pawn = u8x64::splat(0b0001);
        let knight = u8x64::splat(0b0010);
        let bishop = u8x64::splat(0b0100);
        let rook = u8x64::splat(0b1000);

        let packed = pawn_attacks.select(pawn, zeroed())
            | knight_attacks.select(knight, zeroed())
            | bishop_attacks.select(bishop, zeroed())
            | rook_attacks.select(rook, zeroed());

        let unfurled = valid.select(packed.unfurl(rays), zeroed());

        [
            unfurled.bitand(pawn).simd_ne(zeroed()).into(),
            unfurled.bitand(knight).simd_ne(zeroed()).into(),
            unfurled.bitand(bishop).simd_ne(zeroed()).into(),
            unfurled.bitand(rook).simd_ne(zeroed()).into(),
        ]
    }

    /// Computes [`Pins`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pins(&self, threats: &Threats) -> Pins {
        Pins::new(self, threats)
    }

    /// Computes [`Threats`] .
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn threats(&self) -> Threats {
        Threats::new(self)
    }

    /// Computes the [zobrist hashes](`Zobrists`).
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn zobrists(&self) -> Zobrists {
        Zobrists::new(self)
    }

    /// Removes a [`Place`] from a [`Square`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn outplace(&mut self, sq: Square) {
        debug_assert_ne!(self.placement[sq], Place::empty());

        let idx = self.placement[sq].idx().assume();
        let color = self.placement[sq].color().assume();

        self.roles[color][idx] = None;
        self.squares[color][idx] = None;
        self.placement.set(sq, zeroed());
    }

    /// Adds a [`Place`] to a [`Square`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn emplace(&mut self, sq: Square, p: Place) {
        debug_assert_eq!(self.placement[sq], Place::empty());

        let idx = p.idx().assume();
        let color = p.color().assume();

        self.roles[color][idx] = p.role();
        self.squares[color][idx] = Some(sq);
        self.placement.set(sq, p);
    }

    /// Relocates a [`Place`] to a [`Square`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn displace(&mut self, wc: Square, wt: Square, p: Place) {
        debug_assert_ne!(self.placement[wc], Place::empty());
        debug_assert_eq!(self.placement[wt], Place::empty());

        let idx = p.idx().assume();
        let color = p.color().assume();

        self.roles[color][idx] = p.role();
        self.squares[color][idx] = Some(wt);
        self.placement.set(wc, zeroed());
        self.placement.set(wt, p);
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut skip = 0;
        for sq in Square::iter().map(Flip::flip) {
            let mut buffer = [b'\0'; 2];

            if sq.file() == File::H {
                buffer[0] = if sq.rank() == Rank::First { b' ' } else { b'/' };
            }

            match self.placement[sq].piece() {
                None => skip += 1,
                Some(p) => {
                    buffer[1] = buffer[0];
                    write!(&mut buffer[..1], "{p}").assume();
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
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Error)]
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
        let mut board = Board {
            turn: Color::White,
            castles: Default::default(),
            en_passant: Default::default(),
            halfmoves: Default::default(),
            fullmoves: Default::default(),
            placement: Default::default(),
            squares: Default::default(),
            roles: Default::default(),
        };

        let tokens = &mut s.split_ascii_whitespace();
        let Some(placement) = tokens.next() else {
            return Err(ParseFenError::InvalidPlacement);
        };

        let mut idx = [0u8; 2];
        for (rank, segment) in placement.split('/').rev().enumerate() {
            let mut file = 0;
            for c in segment.chars() {
                if file >= 8 || rank >= 8 {
                    return Err(ParseFenError::InvalidPlacement);
                } else if let Some(skip) = c.to_digit(10) {
                    file += skip;
                    continue;
                }

                let mut buffer = [0; 4];
                let sq = Square::new(File::new(file as i8), Rank::new(rank as i8));
                let Ok(p) = Piece::from_str(c.encode_utf8(&mut buffer)) else {
                    return Err(ParseFenError::InvalidPlacement);
                };

                let idx = if p.role() == Role::King {
                    Idx::new(0)
                } else if idx[p.color()] >= Idx::MAX {
                    return Err(ParseFenError::InvalidPlacement);
                } else {
                    idx[p.color()] += 1;
                    Idx::new(idx[p.color()])
                };

                board.emplace(sq, Place::new(p, idx));
                file += 1;
            }
        }

        board.turn = match tokens.next() {
            Some("w") => Color::White,
            Some("b") => Color::Black,
            _ => return Err(ParseFenError::InvalidSideToMove),
        };

        board.castles = match tokens.next() {
            None => return Err(ParseFenError::InvalidCastlingRights),
            Some("-") => Castles::none(),
            Some(s) => match s.parse() {
                Err(_) => return Err(ParseFenError::InvalidCastlingRights),
                Ok(castles) => castles,
            },
        };

        board.en_passant = match tokens.next() {
            None => return Err(ParseFenError::InvalidEnPassantSquare),
            Some("-") => None,
            Some(ep) => match ep.parse() {
                Err(_) => return Err(ParseFenError::InvalidEnPassantSquare),
                Ok(sq) => Some(sq),
            },
        };

        match tokens.next().map(u8::from_str) {
            Some(Ok(halfmoves)) => board.halfmoves = halfmoves,
            _ => return Err(ParseFenError::InvalidHalfmoveClock),
        }

        match tokens.next().map(u32::from_str) {
            Some(Ok(fullmoves)) => board.fullmoves = fullmoves,
            _ => return Err(ParseFenError::InvalidFullmoveNumber),
        }

        if tokens.next().is_some() {
            return Err(ParseFenError::InvalidSyntax);
        }

        Ok(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fmt::Debug, hash::DefaultHasher};
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn board_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Board>>(), size_of::<Board>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn hash_is_consistent(a: Board, b: Board) {
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
    fn by_color_returns_squares_occupied_by_pieces_of_a_color(b: Board, c: Color) {
        for sq in Bitboard::from(b.by_color(c)) {
            assert_eq!(b[sq].color(), Some(c));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn by_role_returns_squares_occupied_by_pieces_of_a_role(b: Board, r: Role) {
        for sq in Bitboard::from(b.by_role(r)) {
            assert_eq!(b[sq].role(), Some(r));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn by_piece_returns_squares_occupied_by_a_piece(b: Board, p: Piece) {
        for sq in Bitboard::from(b.by_piece(p)) {
            assert_eq!(b[sq].piece(), Some(p));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn king_returns_square_occupied_by_a_king(b: Board, c: Color) {
        if let Some(sq) = b.king(c) {
            assert_eq!(b[sq].piece(), Some(Piece::new(Role::King, c)));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn outplace_removes_place_from_square(mut b: Board, #[filter(!#b[#sq].is_empty())] sq: Square) {
        b.outplace(sq);
        assert_eq!(b[sq], Place::empty());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn emplace_adds_place_to_square(
        mut b: Board,
        #[filter(#b[#sq].is_empty())] sq: Square,
        p: Place,
    ) {
        b.emplace(sq, p);
        assert_eq!(b[sq], p);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn displace_relocates_place_to_square(
        mut b: Board,
        #[filter(!#b[#wc].is_empty())] wc: Square,
        #[filter(#b[#wt].is_empty())] wt: Square,
    ) {
        let p = b[wc];

        b.displace(wc, wt, p);
        assert_eq!(b[wc], Place::empty());
        assert_eq!(b[wt], p);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_board_is_an_identity(b: Board) {
        assert_eq!(b.to_string().parse(), Ok(b));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    #[expect(clippy::string_slice)]
    fn parsing_board_fails_for_invalid_placement(
        b: Board,
        #[strategy(..=#b.to_string().len())] n: usize,
        #[strategy("[^[:ascii:]]+")] r: String,
    ) {
        let s = b.to_string();
        assert_eq!([&s[..n], &r, &s[n..]].concat().parse().ok(), None::<Board>);
    }
}
