use crate::chess::{Color, Furl, Idx, IdxSet, Piece, Rays, Role, Square, Unfurl};
use crate::simd::*;
use crate::util::{Assume, Binary, Bits, Int, Num};
use bytemuck::{NoUninit, Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, DerefMut, IntoIterator};
use std::hash::{Hash, Hasher};
use std::{mem::transmute_copy, ops::*};

#[cfg(test)]
use proptest::prelude::*;

/// A place on the board.
#[derive(Debug, Copy, Hash, Zeroable, NoUninit, Deref)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Place(Bits<u8, 8>);

#[cfg(test)]
impl Arbitrary for Place {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        any::<(Piece, Idx)>()
            .prop_map(|(p, idx)| Place::new(p, idx))
            .boxed()
    }
}

const unsafe impl Num for Place {
    type Repr = u8;
    const MIN: Self::Repr = u8::MIN;
    const MAX: Self::Repr = u8::MAX;
}

const impl Place {
    pub const IDX_MASK: u8 = 0b00001111;
    pub const COLOR_MASK: u8 = 0b10000000;
    pub const ROLE_MASK: u8 = 0b01110000;
    pub const PIECE_MASK: u8 = Self::ROLE_MASK | Self::COLOR_MASK;

    #[inline(always)]
    pub fn new(piece: Piece, idx: Idx) -> Self {
        let mut bits = Bits::<u8, 8>::default();
        bits.push(piece.encode());
        bits.push(idx.encode());
        Place(bits)
    }

    #[inline(always)]
    pub fn empty() -> Self {
        zeroed()
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self == Self::empty()
    }

    #[inline(always)]
    pub fn idx(self) -> Option<Idx> {
        if self.is_empty() {
            None
        } else {
            Some(Idx::decode(self.0.slice(..4).convert().assume()))
        }
    }

    #[inline(always)]
    pub fn piece(self) -> Option<Piece> {
        if self.is_empty() {
            None
        } else {
            Some(Piece::decode(self.0.slice(4..).convert().assume()))
        }
    }

    #[inline(always)]
    pub fn color(self) -> Option<Color> {
        if self.is_empty() {
            None
        } else {
            Some(Color::decode(self.0.slice(7..).convert().assume()))
        }
    }

    #[inline(always)]
    pub fn role(self) -> Option<Role> {
        if self.is_empty() {
            None
        } else {
            Some(Role::decode(self.0.slice(4..=6).convert().assume()))
        }
    }
}

const impl Binary for Place {
    type Bits = Bits<u8, 8>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        self.0
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Self(bits)
    }
}

/// The arrangement of [`Places`]s on the [`Board`].
#[derive(Debug, Clone, Copy, Eq, Zeroable, Deref, DerefMut, IntoIterator)]
#[repr(transparent)]
pub struct Placement(
    #[deref(forward)]
    #[deref_mut(forward)]
    #[into_iterator(owned, ref, ref_mut)]
    Aligned<[Place; Square::LEN]>,
);

impl Default for Placement {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl PartialEq for Placement {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.to_simd().simd_eq(other.to_simd()).all()
    }
}

impl Hash for Placement {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_simd().hash(state);
    }
}

impl Placement {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(places: [Place; Square::LEN]) -> Self {
        Self(Aligned(places))
    }

    /// The placement by [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn roles(&self) -> u8x64 {
        (self.to_simd() & Simd::splat(Place::ROLE_MASK)) >> Place::ROLE_MASK.trailing_zeros() as u8
    }

    /// The placement by [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pieces(&self) -> u8x64 {
        self.to_simd() >> Place::PIECE_MASK.trailing_zeros() as u8
    }

    /// The placement by [`Idx`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn indices(&self) -> u8x64 {
        self.to_simd() & Simd::splat(Place::IDX_MASK)
    }

    /// [`Place`]s occupied.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn occupied(&self) -> M8x64 {
        self.to_simd().simd_ne(zeroed()).into()
    }

    /// [`Place`]s vacant.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn vacant(&self) -> M8x64 {
        self.to_simd().simd_eq(zeroed()).into()
    }

    /// [`Place`]s occupied by [`Piece`]s of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_color(&self, c: Color) -> M8x64 {
        let black = self.to_simd().cast::<i8>().is_negative().into();

        match c {
            Color::Black => black,
            Color::White => black ^ self.occupied(),
        }
    }

    /// [`Place`]s occupied by [`Piece`]s of a [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_role(&self, r: Role) -> M8x64 {
        self.roles().simd_eq(Simd::splat(r.encode().get())).into()
    }

    /// [`Place`]s occupied by a [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_piece(&self, p: Piece) -> M8x64 {
        self.pieces().simd_eq(Simd::splat(p.encode().get())).into()
    }

    /// Masks this placement.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask<I: MaskElement>(&self, mask: M<I, 64>) -> Self {
        Self::from_simd(mask.select(self.to_simd(), zeroed()))
    }

    /// Blends a value into this placement.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn blend<I: MaskElement>(&self, mask: M<I, 64>, place: Place) -> Self {
        Self::from_simd(mask.select(Simd::splat(place.get()), self.to_simd()))
    }

    /// Sets a place.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn set(&mut self, sq: Square, place: Place) {
        #[cfg(target_feature = "avx512f")]
        {
            *self = self.blend(M8x64::from(sq.bitboard()), place);
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self[sq] = place;
        }
    }

    /// Convert places to their corresponding [`IdxSet`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_idx_set(&self) -> Wordboard {
        let indices = self.indices();

        #[cfg(target_feature = "avx512f")]
        {
            let ones = self.occupied().select(Simd::splat(1), Simd::splat(0));
            Wordboard::from_simd(ones.shlv(indices.cast::<u16>()))
        }

        #[cfg(not(target_feature = "avx512f"))]
        unsafe {
            #[rustfmt::skip]
            const S0: u8x64 = Simd::from_array([
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]);

            #[rustfmt::skip]
            const S1: u8x64 = Simd::from_array([
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            ]);

            let (left, right) = S0.shuffle(indices).interleave(S1.shuffle(indices));
            let indices = transmute_copy::<[u8x64; 2], u16x64>(&[left, right]);
            Wordboard::from_simd(indices).mask(self.occupied())
        }
    }

    /// Converts to the equivalent simd type by copy.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(&self) -> u8x64 {
        unsafe { transmute_copy::<Self, u8x64>(self) }
    }

    /// Converts from the equivalent simd type.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_simd(simd: u8x64) -> Self {
        unsafe { transmute_copy::<u8x64, Self>(&simd) }
    }
}

impl Furl for Placement {
    type Furled = FurledPlacement;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn furl(&self, rays: Rays) -> Self::Furled {
        FurledPlacement::from_simd(self.to_simd().furl(rays))
    }
}

/// The arrangement of [`Places`]s on the [`Board`] furled in along [`Rays`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Zeroable, Deref, DerefMut)]
#[repr(transparent)]
pub struct FurledPlacement(
    #[deref]
    #[deref_mut]
    Placement,
);

const KING: u8 = 0b0000001;
const WPAWN: u8 = 0b0000010;
const BPAWN: u8 = 0b0000100;
const KNIGHT: u8 = 0b0001000;
const BISHOP: u8 = 0b0010000;
const ROOK: u8 = 0b0100000;
const QUEEN: u8 = 0b1000000;

const DIAG: u8 = BISHOP | QUEEN;
const ORTH: u8 = ROOK | QUEEN;
const OADJ: u8 = ROOK | QUEEN | KING;
const WPDJ: u8 = BISHOP | QUEEN | KING | WPAWN;
const BPDJ: u8 = BISHOP | QUEEN | KING | BPAWN;

#[rustfmt::skip]
static ATTACKERS: u8x64 = u8x64::from_array([
    KNIGHT, OADJ, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
    KNIGHT, WPDJ, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
    KNIGHT, OADJ, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
    KNIGHT, BPDJ, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
    KNIGHT, OADJ, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
    KNIGHT, BPDJ, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
    KNIGHT, OADJ, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
    KNIGHT, WPDJ, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
]);

impl FurledPlacement {
    /// Extends attacks along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn extend(&self) -> Self {
        Self::from_simd(self.to_simd().broadcast1x8())
    }

    /// Masks [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask<I: MaskElement>(&self, mask: M<I, 64>) -> Self {
        Self(self.0.mask(mask))
    }

    /// Splices [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn blend<I: MaskElement>(&self, mask: M<I, 64>, place: Place) -> Self {
        Self(self.0.blend(mask, place))
    }

    /// Visible [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn visible(&self) -> M8x64 {
        #[cfg(target_feature = "avx512f")]
        {
            let o = self.occupied().to_bitmask() | 0x8181818181818181u64;
            M8x64::from_bitmask(o ^ o.wrapping_sub(0x0303030303030303))
        }

        #[cfg(not(target_feature = "avx512f"))]
        unsafe {
            use std::mem::transmute;
            let occ = self.occupied().to_simd();
            let o = transmute::<u64x8, i8x64>(transmute::<i8x64, u64x8>(occ) - u64x8::splat(0x101));
            occ.simd_ne(o).into()
        }
    }

    /// Attacking [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attackers(&self) -> M8x64 {
        #[rustfmt::skip]
        static DECODER: u8x64 = u8x64::from_array([
            0, KING, BPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, WPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, BPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, WPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, BPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, WPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, BPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
            0, KING, WPAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
        ]);

        let pieces = self.to_simd() >> Simd::splat(4);
        let pieces = ATTACKERS & DECODER.shuffle(pieces);
        pieces.simd_ne(zeroed()).into()
    }

    /// Attacking pinners along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pinners(&self) -> M8x64 {
        let mask = ATTACKERS & Simd::splat(0xF0);
        self.to_simd().bitand(mask).simd_eq(mask).into()
    }

    /// Ray attacks for a piece.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attacks(&self, p: Piece) -> M8x64 {
        #[cfg(target_feature = "avx512f")]
        {
            const ATTACKS: [u64; Piece::LEN] = const {
                let mut table: [u64; Piece::LEN] = zeroed();

                for p in Piece::iter() {
                    for sq in Square::iter() {
                        let mask = match p {
                            Piece::WhitePawn => WPAWN,
                            Piece::BlackPawn => BPAWN,
                            Piece::WhiteKnight | Piece::BlackKnight => KNIGHT,
                            Piece::WhiteBishop | Piece::BlackBishop => BISHOP,
                            Piece::WhiteRook | Piece::BlackRook => ROOK,
                            Piece::WhiteQueen | Piece::BlackQueen => QUEEN,
                            Piece::WhiteKing | Piece::BlackKing => KING,
                        };

                        if ATTACKERS.as_array()[sq] & mask != 0 {
                            table[p] |= *sq.bitboard();
                        }
                    }
                }

                table
            };

            ATTACKS[p].into()
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            use {Color::*, Role::*};
            const ATTACKS: [u8x64; Color::LEN] = const {
                let mut table: [u8x64; Color::LEN] = zeroed();

                for c in Color::iter() {
                    let king = 1 << Role::King.get();
                    let white_pawn = if c == White { 1 << Pawn.get() } else { 0 };
                    let black_pawn = if c == Black { 1 << Pawn.get() } else { 0 };
                    let knight = 1 << Knight.get();
                    let bishop = 1 << Bishop.get();
                    let rook = 1 << Rook.get();
                    let queen = 1 << Queen.get();

                    let diag = bishop | queen;
                    let orth = rook | queen;
                    let oadj = rook | queen | king;
                    let wdpj = bishop | queen | king | white_pawn;
                    let bpdj = bishop | queen | king | black_pawn;

                    #[rustfmt::skip]
                    let attacks = u8x64::from_array([
                        knight, oadj, orth, orth, orth, orth, orth, orth,
                        knight, wdpj, diag, diag, diag, diag, diag, diag,
                        knight, oadj, orth, orth, orth, orth, orth, orth,
                        knight, bpdj, diag, diag, diag, diag, diag, diag,
                        knight, oadj, orth, orth, orth, orth, orth, orth,
                        knight, bpdj, diag, diag, diag, diag, diag, diag,
                        knight, oadj, orth, orth, orth, orth, orth, orth,
                        knight, wdpj, diag, diag, diag, diag, diag, diag,
                    ]);

                    table[c] = attacks;
                }

                table
            };

            let bit = Simd::splat(1 << p.role().get());
            ATTACKS[p.color()].bitand(bit).simd_ne(zeroed()).into()
        }
    }

    /// Converts from the equivalent simd type.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_simd(simd: u8x64) -> Self {
        Self(Placement::from_simd(simd))
    }
}

impl Unfurl for FurledPlacement {
    type Unfurled = Placement;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn unfurl(&self, rays: Rays) -> Self::Unfurled {
        Placement::from_simd(self.to_simd().unfurl(rays))
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn unfurl_flip(&self, rays: Rays) -> Self::Unfurled {
        Placement::from_simd(self.to_simd().unfurl_flip(rays))
    }
}

/// [`IdxSet`] for each [`Square`].
#[derive(Debug, Clone, Copy, Eq, Zeroable, Deref, DerefMut, IntoIterator)]
#[repr(transparent)]
pub struct Wordboard(#[into_iterator(owned, ref, ref_mut)] [IdxSet; Square::LEN]);

impl Default for Wordboard {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl PartialEq for Wordboard {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.to_simd().simd_eq(other.to_simd()).all()
    }
}

impl Hash for Wordboard {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_simd().hash(state);
    }
}

impl Wordboard {
    /// [`Squares`]s that contain any [`Idx`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn any(&self) -> M16x64 {
        self.to_simd().simd_ne(zeroed()).into()
    }

    /// [`Squares`] that contain `mask`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn matching(&self, indices: IdxSet) -> M16x64 {
        self.bitand(Simd::splat(*indices)).any()
    }

    /// Masks this vector.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask<I: MaskElement>(&self, mask: M<I, 64>) -> Self {
        Self::from_simd(mask.select(self.to_simd(), zeroed()))
    }

    /// Blends a value into this vector.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn blend<I: MaskElement>(&self, mask: M<I, 64>, indices: IdxSet) -> Self {
        Self::from_simd(mask.select(Simd::splat(indices.get()), self.to_simd()))
    }

    /// Union of all [`IdxSet`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn reduce_or(&self) -> IdxSet {
        IdxSet::new(self.to_simd().reduce_or())
    }

    /// Intersection of all [`IdxSet`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn reduce_and(&self) -> IdxSet {
        IdxSet::new(self.to_simd().reduce_and())
    }

    /// Converts to the equivalent simd type by copy.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(&self) -> u16x64 {
        unsafe { transmute_copy::<Self, u16x64>(self) }
    }

    /// Converts from the equivalent simd type.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_simd(simd: u16x64) -> Self {
        unsafe { transmute_copy::<u16x64, Self>(&simd) }
    }
}

impl Not for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self::from_simd(self.to_simd().not())
    }
}

impl BitAnd for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.bitand(rhs.to_simd())
    }
}

impl BitAnd<u16x64> for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: u16x64) -> Self::Output {
        Self::from_simd(self.to_simd().bitand(rhs))
    }
}

impl BitAndAssign for Wordboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.bitand_assign(rhs.to_simd());
    }
}

impl BitAndAssign<u16x64> for Wordboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: u16x64) {
        *self = self.bitand(rhs);
    }
}

impl BitOr for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.bitor(rhs.to_simd())
    }
}

impl BitOr<u16x64> for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: u16x64) -> Self::Output {
        Self::from_simd(self.to_simd().bitor(rhs))
    }
}

impl BitOrAssign for Wordboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bitor_assign(rhs.to_simd());
    }
}

impl BitOrAssign<u16x64> for Wordboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: u16x64) {
        *self = self.bitor(rhs);
    }
}

impl BitXor for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.bitxor(rhs.to_simd())
    }
}

impl BitXor<u16x64> for Wordboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: u16x64) -> Self::Output {
        Self::from_simd(self.to_simd().bitxor(rhs))
    }
}

impl BitXorAssign for Wordboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.bitxor_assign(rhs.to_simd());
    }
}

impl BitXorAssign<u16x64> for Wordboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: u16x64) {
        *self = self.bitxor(rhs);
    }
}

/// An arrangement of items by [`Idx`].
#[derive(Debug, Default, Clone, Copy, Eq, Deref, DerefMut, IntoIterator)]
#[repr(transparent)]
pub struct ByIdx<T: Num>(#[into_iterator(owned, ref, ref_mut)] [T; Idx::LEN]);

impl<T: Num + PartialEq> PartialEq for ByIdx<T> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.to_simd().simd_eq(other.to_simd()).all()
    }
}

impl<T: Num + Hash> Hash for ByIdx<T> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_simd().hash(state);
    }
}

impl<T: Num> ByIdx<T> {
    #[expect(dead_code)]
    const REQUIRES: () = const { assert!(size_of::<T>() == size_of::<u8>()) };

    /// Blends a value into this arrangement.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn blend<I: MaskElement>(&self, mask: M<I, 16>, item: T) -> Self {
        Self::from_simd(mask.select(Self([item; _]).to_simd(), self.to_simd()))
    }

    /// Sets an item.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn set(&mut self, idx: Idx, item: T) {
        #[cfg(target_feature = "avx512f")]
        {
            *self = self.blend(M8x16::from(idx.to_set()), item);
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self[idx] = item;
        }
    }

    /// Converts to the equivalent simd type by copy.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(&self) -> u8x16 {
        unsafe { transmute_copy::<Self, u8x16>(self) }
    }

    /// Converts from the equivalent simd type.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_simd(simd: u8x16) -> Self {
        unsafe { transmute_copy::<u8x16, Self>(&simd) }
    }
}

impl<T: Num + PartialEq> ByIdx<T> {
    /// Bitmask containing `items`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn containing(&self, item: T) -> M8x16 {
        self.to_simd().simd_eq(Self([item; _]).to_simd()).into()
    }
}

/// The [`Square`] of each piece on the board by [`Color`] and [`Idx`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut, IntoIterator)]
#[repr(transparent)]
pub struct SquareByIdx(#[into_iterator(owned, ref, ref_mut)] [ByIdx<Option<Square>>; Color::LEN]);

impl SquareByIdx {
    #[expect(dead_code)]
    const REQUIRES: () = ByIdx::<Option<Role>>::REQUIRES;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(p: &Placement) -> Self {
        let mut squares = Self::default();

        for sq in Square::iter() {
            if !p[sq].is_empty() {
                let c = p[sq].color().assume();
                let idx = p[sq].idx().assume();
                squares[c][idx] = Some(sq);
            }
        }

        squares
    }
}

/// The [`Role`] of each piece on the board by [`Color`] and [`Idx`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut, IntoIterator)]
#[repr(transparent)]
pub struct RoleByIdx(#[into_iterator(owned, ref, ref_mut)] [ByIdx<Option<Role>>; Color::LEN]);

impl RoleByIdx {
    #[expect(dead_code)]
    const REQUIRES: () = ByIdx::<Option<Role>>::REQUIRES;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(p: &Placement) -> Self {
        let mut roles = Self::default();

        for sq in Square::iter() {
            if !p[sq].is_empty() {
                let c = p[sq].color().assume();
                let idx = p[sq].idx().assume();
                roles[c][idx] = p[sq].role();
            }
        }

        roles
    }
}
