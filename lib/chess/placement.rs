use crate::chess::{Bitboard, Color, Furl, Piece, Role, Square, Unfurl};
use crate::simd::*;
use crate::util::{Assume, Binary, Bits, Int, Num, zero};
use bytemuck::{NoUninit, Pod, Zeroable, zeroed};
use derive_more::with_trait::{Constructor, Debug, Deref, DerefMut, Display};
use std::hash::{Hash, Hasher};
use std::{mem::transmute_copy, ops::*};

#[cfg(test)]
use proptest::prelude::*;

/// A place on the board.
#[derive(Debug, Copy, Hash, Zeroable, NoUninit)]
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
        self.0 == zeroed()
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

/// A numeric identifier for a piece on the board, or none.
#[derive(Debug, Copy, Hash, Zeroable, NoUninit, Constructor)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Idx(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Idx as Num>::Repr);

const unsafe impl Num for Idx {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 15;
}

const unsafe impl Int for Idx {}

const impl Binary for Idx {
    type Bits = Bits<u8, 4>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        self.convert().assume()
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        bits.convert().assume()
    }
}

const impl Idx {
    pub const KING: Self = zero();

    /// This index's [`IdxSet`].
    #[inline(always)]
    pub fn set(self) -> IdxSet {
        IdxSet::new(1 << self.0)
    }
}

const impl<T> Index<Idx> for [T; Idx::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, idx: Idx) -> &Self::Output {
        self.get(idx.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Idx> for [T; Idx::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, idx: Idx) -> &mut Self::Output {
        self.get_mut(idx.cast::<usize>()).assume()
    }
}

/// A set of [`Idx`]s.
#[derive(Debug, Display, Copy, Hash, Zeroable, Pod, Constructor)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("IdxSet({self})")]
#[display("{_0:016b}")]
#[repr(transparent)]
pub struct IdxSet(pub <IdxSet as Num>::Repr);

const unsafe impl Num for IdxSet {
    type Repr = u16;
    const MIN: Self::Repr = u16::MIN;
    const MAX: Self::Repr = u16::MAX;
}

const unsafe impl Int for IdxSet {}

const impl IdxSet {
    /// The number of [`Place`]s in the set.
    #[inline(always)]
    pub fn len(self) -> usize {
        self.0.count_ones().cast::<usize>()
    }

    /// Whether the set is empty.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Whether this [`Idx`] is in the set.
    #[inline(always)]
    pub fn contains(self, idx: Idx) -> bool {
        self & idx.set() != zeroed()
    }

    /// An iterator over the [`Idx`]s in this set.
    #[inline(always)]
    pub fn iter(self) -> Indices {
        Indices::new(self)
    }
}

const impl Deref for IdxSet {
    type Target = u16;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const impl Not for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}

const impl BitAnd for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.bitand(rhs.0)
    }
}

const impl BitAnd<u16> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: u16) -> Self::Output {
        Self(self.0.bitand(rhs))
    }
}

const impl BitAndAssign for IdxSet {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.bitand_assign(rhs.0);
    }
}

const impl BitAndAssign<u16> for IdxSet {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: u16) {
        self.0.bitand_assign(rhs);
    }
}

const impl BitOr for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.bitor(rhs.0)
    }
}

const impl BitOr<u16> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: u16) -> Self::Output {
        Self(self.0.bitor(rhs))
    }
}

const impl BitOrAssign for IdxSet {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bitor_assign(rhs.0);
    }
}

const impl BitOrAssign<u16> for IdxSet {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: u16) {
        self.0.bitor_assign(rhs);
    }
}

const impl BitXor for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.bitxor(rhs.0)
    }
}

const impl BitXor<u16> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: u16) -> Self::Output {
        Self(self.0.bitxor(rhs))
    }
}

const impl BitXorAssign for IdxSet {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.bitxor_assign(rhs.0);
    }
}

const impl BitXorAssign<u16> for IdxSet {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: u16) {
        self.0.bitxor_assign(rhs);
    }
}

const impl IntoIterator for IdxSet {
    type Item = Idx;
    type IntoIter = Indices;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        Indices::new(self)
    }
}

/// An iterator over the [`Idx`]s in an [`IdxSet`].
#[derive(Debug, Constructor)]
pub struct Indices(IdxSet);

const impl Indices {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

const impl Iterator for Indices {
    type Item = Idx;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            None
        } else {
            let idx: Idx = self.0.trailing_zeros().convert().assume();
            self.0 ^= idx.set();
            Some(idx)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for Indices {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

/// The arrangement of [`Places`]s on the [`Board`].
#[derive(Debug, Clone, Copy, Eq, Zeroable, Deref, DerefMut)]
#[repr(transparent)]
pub struct Placement(
    #[deref(forward)]
    #[deref_mut(forward)]
    Aligned<[Place; Square::MAX as usize + 1]>,
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
    pub fn new(places: [Place; Square::MAX as usize + 1]) -> Self {
        Self(Aligned(places))
    }

    /// The placement by [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn roles(&self) -> u8x64 {
        self.to_simd() & Simd::splat(Place::ROLE_MASK)
    }

    /// The placement by [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pieces(&self) -> u8x64 {
        self.to_simd() & Simd::splat(Place::PIECE_MASK)
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
    pub fn occupied(&self) -> Bitboard {
        self.to_simd().simd_ne(zeroed()).into()
    }

    /// [`Place`]s occupied by [`Piece`]s of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_color(&self, c: Color) -> Bitboard {
        let black = self.to_simd().cast::<i8>().is_negative().into();

        match c {
            Color::Black => black,
            Color::White => black ^ self.occupied(),
        }
    }

    /// [`Place`]s occupied by [`Piece`]s of a [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_role(&self, r: Role) -> Bitboard {
        let target = Simd::splat(r.encode().get() << Place::ROLE_MASK.trailing_zeros());
        self.roles().simd_eq(target).into()
    }

    /// [`Place`]s occupied by a [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_piece(&self, p: Piece) -> Bitboard {
        let target = Simd::splat(p.encode().get() << Place::PIECE_MASK.trailing_zeros());
        self.pieces().simd_eq(target).into()
    }

    /// Masks this placement.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask(&self, mask: Bitboard) -> Self {
        Self::from_simd(mask.select(self.to_simd(), zeroed()))
    }

    /// Splices this placement.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn splice(&self, mask: Bitboard, place: Place) -> Self {
        let places = Self::new([place; _]).to_simd();
        Self::from_simd(mask.select(places, self.to_simd()))
    }

    /// Converts to the equivalent simd type by copy.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(&self) -> u8x64 {
        self.0.cast()
    }

    /// Converts from the equivalent simd type.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_simd(simd: u8x64) -> Self {
        unsafe { transmute_copy::<u8x64, Self>(&simd) }
    }
}

impl BitAnd for Placement {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.bitand(rhs.to_simd())
    }
}

impl BitAnd<u8x64> for Placement {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: u8x64) -> Self::Output {
        Self::from_simd(self.to_simd().bitand(rhs))
    }
}

impl BitAndAssign for Placement {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.bitand_assign(rhs.to_simd());
    }
}

impl BitAndAssign<u8x64> for Placement {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: u8x64) {
        *self = self.bitand(rhs);
    }
}

impl BitOr for Placement {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.bitor(rhs.to_simd())
    }
}

impl BitOr<u8x64> for Placement {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: u8x64) -> Self::Output {
        Self::from_simd(self.to_simd().bitor(rhs))
    }
}

impl BitOrAssign for Placement {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bitor_assign(rhs.to_simd());
    }
}

impl BitOrAssign<u8x64> for Placement {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: u8x64) {
        *self = self.bitor(rhs);
    }
}

impl Furl for Placement {
    type Furled = FurledPlacement;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn furl(&self, rays: u8x64) -> Self::Furled {
        FurledPlacement::from_simd(self.to_simd().furl(rays))
    }
}

/// The arrangement of [`Places`]s on the [`Board`] furled in along [`Rays`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Zeroable, Deref, DerefMut)]
#[repr(transparent)]
pub struct FurledPlacement(
    #[deref]
    #[deref_mut]
    Placement,
);

impl Default for FurledPlacement {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

const KING: u8 = 1 << 0;
const WHITE_PAWN: u8 = 1 << 1;
const BLACK_PAWN: u8 = 1 << 2;
const KNIGHT: u8 = 1 << 3;
const BISHOP: u8 = 1 << 4;
const ROOK: u8 = 1 << 5;
const QUEEN: u8 = 1 << 6;

const DIAG: u8 = BISHOP | QUEEN;
const ORTH: u8 = ROOK | QUEEN;
const OADJ: u8 = ROOK | QUEEN | KING;
const WPDJ: u8 = BISHOP | QUEEN | KING | WHITE_PAWN;
const BPDJ: u8 = BISHOP | QUEEN | KING | BLACK_PAWN;

#[rustfmt::skip]
static BITS_TO_ROLE: u8x64 = u8x64::from_array([
    0, KING, BLACK_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, WHITE_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, BLACK_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, WHITE_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, BLACK_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, WHITE_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, BLACK_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
    0, KING, WHITE_PAWN, KNIGHT, 0, BISHOP, ROOK, QUEEN,
]);

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
    /// [`Place`]s along [`Rays`] occupied.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn occupied(&self) -> Bitboard {
        self.0.occupied()
    }

    /// [`Place`]s along [`Rays`] occupied by [`Piece`]s of a [`Color`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_color(&self, c: Color) -> Bitboard {
        self.0.by_color(c)
    }

    /// [`Place`]s along [`Rays`] occupied by [`Piece`]s of a [`Role`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_role(&self, r: Role) -> Bitboard {
        self.0.by_role(r)
    }

    /// [`Place`]s occupied by a [`Piece`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn by_piece(&self, p: Piece) -> Bitboard {
        self.0.by_piece(p)
    }

    /// Extends attacks along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn extend(&self) -> Self {
        Self::from_simd(self.to_simd().broadcast1x8())
    }

    /// Masks [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask(&self, mask: Bitboard) -> Self {
        FurledPlacement(self.0.mask(mask))
    }

    /// Splices [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn splice(&self, mask: Bitboard, place: Place) -> Self {
        FurledPlacement(self.0.splice(mask, place))
    }

    /// Visible [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn visible(&self) -> Bitboard {
        self.beyond(Bitboard::empty())
    }

    /// Visible [`Place`]s along [`Rays`] beyond `blockers`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn beyond(&self, blockers: Bitboard) -> Bitboard {
        let o = (self.occupied() & !blockers) | 0x8181818181818181u64;
        o ^ o.wrapping_sub(0x0303030303030303)
    }

    /// Attacking [`Place`]s along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attackers(&self) -> Bitboard {
        let pieces = self.to_simd() >> Simd::splat(4);
        let pieces = ATTACKERS & BITS_TO_ROLE.shuffle(pieces);
        pieces.simd_ne(zeroed()).into()
    }

    /// Attacking pinners along [`Rays`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn pinners(&self) -> Bitboard {
        let mask = ATTACKERS & Simd::splat(0xF0);
        self.to_simd().bitand(mask).simd_eq(mask).into()
    }

    /// Ray attacks for a piece.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attacks(&self, p: Piece) -> Bitboard {
        const ATTACKS: [Bitboard; Piece::MAX as usize + 1] = const {
            let mut table: [Bitboard; Piece::MAX as usize + 1] = zeroed();

            for p in Piece::iter() {
                for sq in Square::iter() {
                    let mask = match p {
                        Piece::WhitePawn => WHITE_PAWN,
                        Piece::BlackPawn => BLACK_PAWN,
                        Piece::WhiteKnight | Piece::BlackKnight => KNIGHT,
                        Piece::WhiteBishop | Piece::BlackBishop => BISHOP,
                        Piece::WhiteRook | Piece::BlackRook => ROOK,
                        Piece::WhiteQueen | Piece::BlackQueen => QUEEN,
                        Piece::WhiteKing | Piece::BlackKing => KING,
                    };

                    if ATTACKERS.as_array()[sq] & mask != 0 {
                        table[p] |= sq.bitboard();
                    }
                }
            }

            table
        };

        ATTACKS[p]
    }

    /// Converts to the equivalent simd type by copy.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(&self) -> u8x64 {
        self.0.to_simd()
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
    fn unfurl(&self, rays: u8x64) -> Self::Unfurled {
        Placement::from_simd(self.to_simd().unfurl(rays))
    }
}

/// [`IdxSet`] for each [`Square`].
#[derive(Debug, Clone, Copy, Eq, Zeroable, Deref, DerefMut)]
#[repr(transparent)]
pub struct Wordboard([IdxSet; Square::MAX as usize + 1]);

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
    /// [`Squares`] that contain `mask`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn matching(&self, mask: IdxSet) -> Bitboard {
        let masked = self.to_simd().bitand(Simd::splat(*mask));
        masked.simd_ne(zeroed()).to_bitmask().convert().assume()
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

/// Information by each piece on the board by [`Idx`].
///
/// Requires `size_of::<T> == size_of::<u8>()`.
#[derive(Debug, Default, Clone, Copy, Eq, Constructor, Deref, DerefMut)]
#[repr(transparent)]
pub struct ByIdx<T: Copy>([T; Idx::MAX as usize + 1]);

impl<T: Copy + PartialEq> PartialEq for ByIdx<T> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.to_simd().simd_eq(other.to_simd()).all()
    }
}

impl<T: Copy + Hash> Hash for ByIdx<T> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_simd().hash(state);
    }
}

impl<T: Copy> ByIdx<T> {
    /// Converts to the equivalent simd type by copy.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(&self) -> u8x16 {
        const { assert!(size_of::<T>() == size_of::<u8>()) }
        unsafe { transmute_copy::<Self, u8x16>(self) }
    }

    /// Converts from the equivalent simd type.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_simd(simd: u8x16) -> Self {
        const { assert!(size_of::<T>() == size_of::<u8>()) }
        unsafe { transmute_copy::<u8x16, Self>(&simd) }
    }
}

impl<T: Copy + PartialEq> ByIdx<T> {
    /// Bitmask matching `items`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn matching(&self, item: T) -> IdxSet {
        let items = Self::new([item; _]).to_simd();
        let bitmask = self.to_simd().simd_eq(items).to_bitmask();
        bitmask.convert().assume()
    }
}

/// The [`Square`] of each piece on the board by [`Color`] and [`Idx`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
#[repr(transparent)]
pub struct SquareByIdx([ByIdx<Option<Square>>; Color::MAX as usize + 1]);

impl SquareByIdx {
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
#[repr(transparent)]
pub struct RoleByIdx([ByIdx<Option<Role>>; Color::MAX as usize + 1]);

impl RoleByIdx {
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
