use crate::util::{Assume, Int, Num, ones};
use crate::{chess::*, simd::*};
use bytemuck::zeroed;
use std::array;
use std::ops::{BitAnd, Index, IndexMut, Not};

/// A piece-square feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PSQFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PSQFeature as Num>::Repr);

const unsafe impl Num for PSQFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for PSQFeature {}

impl PSQFeature {
    /// The total number of different piece-square features.
    pub const LEN: usize = Square::LEN * Piece::LEN;

    /// Constructs a lookup table for [`PSQFeature`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lut(
        side: Color,
        ksq: Square,
        placement: &Placement,
    ) -> Simd<<Self as Num>::Repr, { Square::LEN }> {
        let pieces = Piece::DECODER.shuffle(placement.pieces()) ^ Simd::splat(side.get());

        let perspective = Square::A1.perspective(side);
        let chirality = Square::A1.perspective(Side::from(ksq.file() < File::E));
        let orientation = Simd::splat(perspective.cast::<u8>() | chirality.cast::<u8>());
        let squares = u8x64::from_array(array::from_fn(Num::cast)) ^ orientation;

        u16x64::splat(Square::LEN.cast()) * pieces.cast::<u16>() + squares.cast::<u16>()
    }
}

impl<T> Index<PSQFeature> for [T; PSQFeature::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: PSQFeature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

impl<T> IndexMut<PSQFeature> for [T; PSQFeature::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, f: PSQFeature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}

/// The king's bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct KingBucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <KingBucket as Num>::Repr);

const unsafe impl Num for KingBucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 31;
}

const unsafe impl Int for KingBucket {}

impl KingBucket {
    pub const LEN: usize = Self::MAX as usize + 1;

    #[inline(always)]
    pub fn new(side: Color, ksq: Square) -> Self {
        #[rustfmt::skip]
        const BUCKETS: [u8; 64] = [
            16, 17, 18, 19,  3,  2,  1,  0,
            20, 21, 22, 23,  7,  6,  5,  4,
            24, 25, 26, 27, 11, 10,  9,  8,
            24, 25, 26, 27, 11, 10,  9,  8,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
        ];

        Num::new(BUCKETS[ksq.perspective(side)])
    }
}

impl<T> Index<KingBucket> for [T; KingBucket::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, b: KingBucket) -> &Self::Output {
        self.get(b.cast::<usize>()).assume()
    }
}

impl<T> IndexMut<KingBucket> for [T; KingBucket::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, b: KingBucket) -> &mut Self::Output {
        self.get_mut(b.cast::<usize>()).assume()
    }
}

/// A king-bucketed PSQ feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct KAFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <KAFeature as Num>::Repr);

const unsafe impl Num for KAFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for KAFeature {}

impl KAFeature {
    /// The total number of different king-piece-square features.
    pub const LEN: usize = PSQFeature::LEN * KingBucket::LEN / 2;

    /// Constructs a lookup table for [`KAFeature`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lut(
        side: Color,
        ksq: Square,
        placement: &Placement,
    ) -> Simd<<Self as Num>::Repr, { Square::LEN }> {
        let chirality = Side::from(ksq.file() < File::E);
        let bucket = u16x64::splat(KingBucket::new(side, ksq.perspective(chirality)).cast());
        u16x64::splat(PSQFeature::LEN.cast()) * bucket + PSQFeature::lut(side, ksq, placement)
    }
}

impl<T> Index<KAFeature> for [T; KAFeature::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: KAFeature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

impl<T> IndexMut<KAFeature> for [T; KAFeature::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, f: KAFeature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}

/// A threat feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct TIFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <TIFeature as Num>::Repr);

const unsafe impl Num for TIFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for TIFeature {}

const fn threat_index_for(role: Role) -> [u16; 65] {
    let mut table = [0u16; 65];

    let mut count = 0;
    for sq in Square::iter() {
        table[sq as usize] = count;
        count += Piece::new(role, Color::White).attacks(sq).len() as u16;
    }

    table[64] = count;
    table
}

impl TIFeature {
    const PAWN_INDEX: u16 = 84;
    const KNIGHT_INDICES: [u16; 65] = threat_index_for(Role::Knight);
    const BISHOP_INDICES: [u16; 65] = threat_index_for(Role::Bishop);
    const ROOK_INDICES: [u16; 65] = threat_index_for(Role::Rook);
    const QUEEN_INDICES: [u16; 65] = threat_index_for(Role::Queen);

    const KNIGHT_OFFSET: u16 = 4 * Self::PAWN_INDEX;
    const BISHOP_OFFSET: u16 = Self::KNIGHT_OFFSET + 10 * Self::KNIGHT_INDICES[64];
    const ROOK_OFFSET: u16 = Self::BISHOP_OFFSET + 8 * Self::BISHOP_INDICES[64];
    const QUEEN_OFFSET: u16 = Self::ROOK_OFFSET + 8 * Self::ROOK_INDICES[64];
    const THREAT_FEATURES: u16 = Self::QUEEN_OFFSET + 10 * Self::QUEEN_INDICES[64];

    /// The total number of different threat features.
    pub const LEN: usize = 2 * Self::THREAT_FEATURES as usize;

    /// Constructs a [`ThreatFeature`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(
        side: Color,
        ksq: Square,
        src: Piece,
        wc: Square,
        dst: Piece,
        wt: Square,
    ) -> Option<Self> {
        let chirality = Side::from(ksq.file() < File::E);
        let wc = wc.perspective(side).perspective(chirality);
        let wt = wt.perspective(side).perspective(chirality);
        let src = src.perspective(side);
        let dst = dst.perspective(side);

        let idx = match src.role() {
            Role::Pawn => Self::pawn_threat_idx(src, wc, dst, wt),
            Role::Knight => Self::knight_threat_idx(src, wc, dst, wt),
            Role::Bishop => Self::bishop_threat_idx(src, wc, dst, wt),
            Role::Rook => Self::rook_threat_idx(src, wc, dst, wt),
            Role::Queen => Self::queen_threat_idx(src, wc, dst, wt),
            Role::King => None,
        }?;

        Some(Num::new(
            idx + src.color().cast::<u16>() * Self::THREAT_FEATURES,
        ))
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn pawn_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        let stride = Self::stride(src, dst)?;
        let rank = wc.rank().cast::<u16>() - 1;
        let file = wc.file().cast::<u16>();
        let leftward = wt.file() < wc.file();
        let attack = 2 * file - u16::from(leftward);
        Some(stride * Self::PAWN_INDEX + rank * 14 + attack)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn knight_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::KNIGHT_OFFSET, &Self::KNIGHT_INDICES)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bishop_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::BISHOP_OFFSET, &Self::BISHOP_INDICES)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn rook_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::ROOK_OFFSET, &Self::ROOK_INDICES)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn queen_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::QUEEN_OFFSET, &Self::QUEEN_INDICES)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn piece_threat_idx(
        src: Piece,
        wc: Square,
        dst: Piece,
        wt: Square,
        offset: u16,
        indices: &[u16; 65],
    ) -> Option<u16> {
        if wt > wc && src.role() == dst.role() {
            return None;
        }

        let stride = Self::stride(src, dst)?;
        let below = src.attacks(wc).bitand(ones::<u64>(wt.cast())).len();
        Some(offset + stride * indices[64] + indices[wc.cast::<usize>()] + below.cast::<u16>())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn stride(src: Piece, dst: Piece) -> Option<u16> {
        const P: [[Option<u16>; 6]; 2] = [
            [None, Some(0), None, Some(1), None, None],
            [None, Some(2), None, Some(3), None, None],
        ];

        const KQ: [[Option<u16>; 6]; 2] = [
            [Some(0), Some(1), Some(2), Some(3), Some(4), None],
            [Some(5), Some(6), Some(7), Some(8), Some(9), None],
        ];

        const BR: [[Option<u16>; 6]; 2] = [
            [Some(0), Some(1), Some(2), Some(3), None, None],
            [Some(4), Some(5), Some(6), Some(7), None, None],
        ];

        match src.role() {
            Role::Pawn => P[dst.color()][dst.role()],
            Role::Knight | Role::Queen => KQ[dst.color()][dst.role()],
            Role::Bishop | Role::Rook => BR[dst.color()][dst.role()],
            Role::King => None,
        }
    }
}

/// A pawn feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PFeature as Num>::Repr);

const unsafe impl Num for PFeature {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for PFeature {}

impl PFeature {
    const SQUARES: u8 = 48;

    /// The total number of different pawn features.
    pub const LEN: usize = 2 * Self::SQUARES.cast::<usize>();

    /// Constructs a lookup table for [`PFeature`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lut(
        side: Color,
        ksq: Square,
        placement: &Placement,
    ) -> Simd<<Self as Num>::Repr, { Square::LEN }> {
        let blacks = placement.by_color(Color::Black);
        let colors = blacks.select(Simd::splat(side.not().get()), Simd::splat(side.get()));

        let perspective = Square::A1.perspective(side);
        let chirality = Square::A1.perspective(Side::from(ksq.file() < File::E));
        let orient = Simd::splat(perspective.cast::<u8>() | chirality.cast::<u8>());
        let squares = u8x64::from_array(array::from_fn(Num::cast)) ^ orient;
        Simd::splat(Self::SQUARES) * colors + squares - Simd::splat(8)
    }
}

impl<T> Index<PFeature> for [T; PFeature::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: PFeature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

impl<T> IndexMut<PFeature> for [T; PFeature::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, f: PFeature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}

/// A pawn-pawn feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PPFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PPFeature as Num>::Repr);

const unsafe impl Num for PPFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for PPFeature {}

impl PPFeature {
    /// The total number of different pawn-pawn features.
    pub const LEN: usize = PFeature::LEN * (PFeature::LEN - 1) / 2;

    /// A mask for pawns visible from a [`File`].
    pub const WINDOW: [Bitboard; 8] = [
        File::A.bitboard() | File::B.bitboard(),
        File::A.bitboard() | File::B.bitboard() | File::C.bitboard(),
        File::B.bitboard() | File::C.bitboard() | File::D.bitboard(),
        File::C.bitboard() | File::D.bitboard() | File::E.bitboard(),
        File::D.bitboard() | File::E.bitboard() | File::F.bitboard(),
        File::E.bitboard() | File::F.bitboard() | File::G.bitboard(),
        File::F.bitboard() | File::G.bitboard() | File::H.bitboard(),
        File::G.bitboard() | File::H.bitboard(),
    ];

    /// Constructs a [`PPFeature`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(ft1: PFeature, ft2: PFeature) -> Self {
        let hi = ft1.max(ft2).cast::<u16>();
        let lo = ft1.min(ft2).cast::<u16>();
        Num::new(lo + hi * (hi - 1) / 2)
    }

    /// Constructs a lookup table for [`PPFeature`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lut(pfts: u8x64, s: Square) -> Simd<<Self as Num>::Repr, 32> {
        const INDICES: [u8x64; 8] = const {
            let mut indices: [u8x64; 8] = zeroed();
            for f in File::iter() {
                let mut i = 0;

                #[allow(clippy::explicit_counter_loop)]
                for sq in PPFeature::WINDOW[f].transpose() {
                    assert!(i < 32);
                    indices[f].as_mut_array()[i] = sq.transpose().cast();
                    i += 1;
                }
            }

            indices
        };

        let pft1 = u8x32::splat(pfts.as_array()[s]);
        let pft2 = pfts.permute(INDICES[s.file()]).extract::<0, 32>();
        let hi = pft1.simd_max(pft2).cast::<<Self as Num>::Repr>();
        let lo = pft1.simd_min(pft2).cast::<<Self as Num>::Repr>();
        lo + ((hi * (hi - Simd::splat(1))) >> 1)
    }
}

impl<T> Index<PPFeature> for [T; PPFeature::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: PPFeature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

impl<T> IndexMut<PPFeature> for [T; PPFeature::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, f: PPFeature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}
