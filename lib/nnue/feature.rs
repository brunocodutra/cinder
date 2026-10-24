use crate::util::{Assume, Int, Num, ones};
use crate::{chess::*, nnue::arch, simd::*};
use bytemuck::{Zeroable, zeroed};
use std::array;
use std::ops::{Index, IndexMut, Not};

/// A piece-square feature with merged kings and horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PSQFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PSQFeature as Num>::Repr);

const unsafe impl Num for PSQFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Self::LEN - 1).cast();
}

const unsafe impl Int for PSQFeature {}

impl PSQFeature {
    /// The total number of different piece-square features.
    pub const LEN: usize = Square::LEN * (Piece::LEN - 1);
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct KingBucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <KingBucket as Num>::Repr);

const unsafe impl Num for KingBucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Self::LEN - 1).cast();
}

const unsafe impl Int for KingBucket {}

impl KingBucket {
    pub const LEN: usize = arch::BUCKETS;

    #[inline(always)]
    pub fn new(side: Color, ksq: Square) -> Self {
        #[rustfmt::skip]
        const BUCKETS: [u8; KingBucket::LEN] = [
            32, 33, 34, 35,  3,  2,  1,  0,
            36, 37, 38, 39,  7,  6,  5,  4,
            40, 41, 42, 43, 11, 10,  9,  8,
            44, 45, 46, 47, 15, 14, 13, 12,
            48, 49, 50, 51, 19, 18, 17, 16,
            52, 53, 54, 55, 23, 22, 21, 20,
            56, 57, 58, 59, 27, 26, 25, 24,
            60, 61, 62, 63, 31, 30, 29, 28,
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

/// A king-bucketed PSQ feature with merged kings and horizontal mirroring..
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct KAFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <KAFeature as Num>::Repr);

const unsafe impl Num for KAFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Self::LEN - 1).cast();
}

const unsafe impl Int for KAFeature {}

impl KAFeature {
    #[expect(dead_code)]
    const REQUIRES: () = const { assert!(Self::LEN == arch::KA) };

    /// The total number of different king-piece-square features.
    pub const LEN: usize = PSQFeature::LEN * KingBucket::LEN / 2;

    /// The maximum number of active king-piece-square features per perspective.
    pub const MAX_ACTIVE: usize = 32;

    /// Compresses the squares and pieces enabled by `mask`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn compress(pieces: u8x64, mask: M8x64) -> (u8x32, u8x32) {
        let mask = mask.to_bitmask();
        let squares = u8x64::from_array(array::from_fn(Num::cast));
        let [squares, _] = squares.compress(mask).halve();
        let [pieces, _] = pieces.compress(mask).halve();
        (squares, pieces)
    }

    /// Constructs a lookup table for compressed [`KAFeature`]s.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lut(
        side: Color,
        ksq: Square,
        pieces: u8x32,
        mut squares: u8x32,
    ) -> Simd<<Self as Num>::Repr, { Self::MAX_ACTIVE }> {
        /// Decodes a placement piece, flips perspective, and merges both kings.
        const DECODER: [u8x32; Color::LEN] = const {
            let mut decoders: [u8x32; Color::LEN] = zeroed();

            for side in Color::iter() {
                for piece in Piece::iter() {
                    let decoded = if piece.role() != Role::King {
                        piece.perspective(side)
                    } else {
                        Piece::WhiteKing
                    };

                    decoders[side].as_mut_array()[piece.cast::<usize>()] = decoded.cast();
                    decoders[side].as_mut_array()[piece.cast::<usize>() + 16] = decoded.cast();
                }
            }

            decoders
        };

        let chirality = Side::from(ksq.file() < File::E);
        let bucket = u16x32::splat(KingBucket::new(side, ksq.perspective(chirality)).cast());
        let pieces = DECODER[side].shuffle(pieces).cast::<u16>();

        let side = Square::A1.perspective(side);
        let chirality = Square::A1.perspective(chirality);
        squares ^= u8x32::splat(side.cast::<u8>() | chirality.cast::<u8>());

        u16x32::splat(PSQFeature::LEN.cast()) * bucket
            + u16x32::splat(Square::LEN.cast()) * pieces
            + squares.cast::<u16>()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct TIFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <TIFeature as Num>::Repr);

const unsafe impl Num for TIFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Self::LEN - 1).cast();
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
    #[expect(dead_code)]
    const REQUIRES: () = const {
        assert!(Self::LEN == arch::TI);
        assert!(Self::LEN <= u16::MAX as usize);
        assert!(Self::LEN < Self::SENTINEL as usize);
    };

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

    const SENTINEL: u16 = 0xF000;
    const OFFSET: [[u16; Square::LEN]; Piece::LEN] = const {
        let mut table = [[0u16; Square::LEN]; Piece::LEN];

        for src in Piece::iter() {
            let indices = threat_index_for(src.role());

            for wc in Square::iter() {
                table[src][wc] = indices[wc.cast::<usize>()];
            }
        }

        table
    };

    const BASE: [[[u16; 2]; Piece::LEN]; Piece::LEN] = const {
        let mut table = [[[0u16; 2]; Piece::LEN]; Piece::LEN];

        for src in Piece::iter() {
            let indices = threat_index_for(src.role());
            let color = src.color().cast::<u16>() * Self::THREAT_FEATURES;
            let role = match src.role() {
                Role::Pawn | Role::King => 0,
                Role::Knight => Self::KNIGHT_OFFSET,
                Role::Bishop => Self::BISHOP_OFFSET,
                Role::Rook => Self::ROOK_OFFSET,
                Role::Queen => Self::QUEEN_OFFSET,
            };

            for dst in Piece::iter() {
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

                let stride = match src.role() {
                    Role::Pawn => P[dst.color()][dst.role()],
                    Role::Knight | Role::Queen => KQ[dst.color()][dst.role()],
                    Role::Bishop | Role::Rook => BR[dst.color()][dst.role()],
                    Role::King => None,
                };

                for semi in 0..2 {
                    let valid = semi == 0 || src.role() != dst.role();
                    table[src][dst][semi] = match (stride, valid) {
                        (Some(stride), true) => color + role + stride * indices[64],
                        _ => Self::SENTINEL,
                    };
                }
            }
        }

        table
    };

    /// The total number of different threat features.
    pub const LEN: usize = 2 * Self::THREAT_FEATURES as usize;

    /// The maximum number of active threat features per perspective.
    pub const MAX_ACTIVE: usize = 128;

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

        let attacks = src.attacks(wc) & ones::<u64>(wt.cast());
        let idx = Self::BASE[src][dst][usize::from(wt > wc)]
            + Self::OFFSET[src][wc]
            + attacks.len().cast::<u16>();

        idx.convert()
    }
}

/// A pawn feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PFeature as Num>::Repr);

const unsafe impl Num for PFeature {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Self::LEN - 1).cast();
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PPFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PPFeature as Num>::Repr);

const unsafe impl Num for PPFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Self::LEN - 1).cast();
}

const unsafe impl Int for PPFeature {}

impl PPFeature {
    #[expect(dead_code)]
    const REQUIRES: () = const { assert!(Self::LEN == arch::PP) };

    /// The total number of different pawn-pawn features.
    pub const LEN: usize = PFeature::LEN * (PFeature::LEN - 1) / 2;

    /// The maximum number of active pawn-pawn features per perspective.
    pub const MAX_ACTIVE: usize = 96;

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

                #[expect(clippy::explicit_counter_loop)]
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
