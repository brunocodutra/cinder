use crate::syzygy::{Dtz, MAX_PIECES, Material, RandomAccessFile, Wdl};
use crate::util::{Assume, Bits, Int};
use crate::{chess::*, syzygy::NormalizedMaterial};
use arrayvec::ArrayVec;
use byteorder::{BE, ByteOrder, LE, ReadBytesExt};
use derive_more::with_trait::Deref;
use std::{hint::unreachable_unchecked, io, marker::PhantomData, path::Path};

/// Metric stored in a table: WDL or DTZ.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Metric {
    /// WDL<sub>50</sub>.
    Wdl,
    /// DTZ<sub>50</sub>.
    Dtz,
}

pub trait TableDescriptor {
    /// One of [`Metric::Wdl`] or [`Metric::Dtz`].
    const METRIC: Metric;
    /// File extension, i.e. `rtbw` or `rtbz`.
    const EXTENSION: &'static str;
    /// Magic header bytes.
    const MAGIC: [u8; 4];
}

impl TableDescriptor for Wdl {
    const METRIC: Metric = Metric::Wdl;
    const MAGIC: [u8; 4] = [0x71, 0xE8, 0x23, 0x5D];
    const EXTENSION: &'static str = "rtbw";
}

impl TableDescriptor for Dtz {
    const METRIC: Metric = Metric::Dtz;
    const MAGIC: [u8; 4] = [0xD7, 0x66, 0x0C, 0xA5];
    const EXTENSION: &'static str = "rtbz";
}

/// Table layout flags.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Deref)]
#[repr(transparent)]
struct Layout(Bits<u8, 8>);

unsafe impl Int for Layout {
    type Repr = u8;
}

impl Layout {
    /// Two sided table for non-symmetrical material configuration.
    const SPLIT: Self = Layout(Bits::new(0b01));
    /// Table with pawns. Has sub-tables for each leading pawn file (a-d).
    const HAS_PAWNS: Self = Layout(Bits::new(0b10));
}

/// Sub-table format flags.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Deref)]
#[repr(transparent)]
struct Flag(Bits<u8, 8>);

unsafe impl Int for Flag {
    type Repr = u8;
}

impl Flag {
    /// DTZ table stores black to move.
    const STM: Self = Self(Bits::new(0b00000001));
    /// Use `DtzMap`.
    const MAPPED: Self = Self(Bits::new(0b00000010));
    /// DTZ table has winning positions on the edge of the 50-move rule and
    /// therefore stores exact plies rather than just full moves.
    const WIN_PLIES: Self = Self(Bits::new(0b00000100));
    /// DTZ table has losing positions on the edge of the 50-move rule and
    /// therefore stores exact plies rather than just full moves.
    const LOSS_PLIES: Self = Self(Bits::new(0b00001000));
    /// DTZ table contains very long endgames, so that values require 16
    /// bits rather than just 8.
    const WIDE_DTZ: Self = Self(Bits::new(0b00010000));
    /// Table stores only a single value.
    const SINGLE_VALUE: Self = Self(Bits::new(0b10000000));
}

/// Maximum size in bytes of a compressed block.
const MAX_BLOCK_SIZE: usize = 1024;

/// Maps squares into the a1-d1-d4 triangle.
#[rustfmt::skip]
const TRIANGLE: [u64; 64] = [
    6, 0, 1, 2, 2, 1, 0, 6,
    0, 7, 3, 4, 4, 3, 7, 0,
    1, 3, 8, 5, 5, 8, 3, 1,
    2, 4, 5, 9, 9, 5, 4, 2,
    2, 4, 5, 9, 9, 5, 4, 2,
    1, 3, 8, 5, 5, 8, 3, 1,
    0, 7, 3, 4, 4, 3, 7, 0,
    6, 0, 1, 2, 2, 1, 0, 6,
];

/// Inverse of `TRIANGLE`.
const INV_TRIANGLE: [usize; 10] = [1, 2, 3, 10, 11, 19, 0, 9, 18, 27];

/// Maps the b1-h1-h7 triangle to `0..=27`.
#[rustfmt::skip]
const LOWER: [u64; 64] = [
    28,  0,  1,  2,  3,  4,  5,  6,
     0, 29,  7,  8,  9, 10, 11, 12,
     1,  7, 30, 13, 14, 15, 16, 17,
     2,  8, 13, 31, 18, 19, 20, 21,
     3,  9, 14, 18, 32, 22, 23, 24,
     4, 10, 15, 19, 22, 33, 25, 26,
     5, 11, 16, 20, 23, 25, 34, 27,
     6, 12, 17, 21, 24, 26, 27, 35,
];

/// Used to initialize `Consts::mult_idx` and `Consts::mult_factor`.
#[rustfmt::skip]
const MULT_TWIST: [u64; 64] = [
    15, 63, 55, 47, 40, 48, 56, 12,
    62, 11, 39, 31, 24, 32,  8, 57,
    54, 38,  7, 23, 16,  4, 33, 49,
    46, 30, 22,  3,  0, 17, 25, 41,
    45, 29, 21,  2,  1, 18, 26, 42,
    53, 37,  6, 20, 19,  5, 34, 50,
    61, 10, 36, 28, 27, 35,  9, 58,
    14, 60, 52, 44, 43, 51, 59, 13,
];

/// Unused entry.
const Z0: u64 = u64::MAX;

/// Encoding of all 462 configurations of two not-connected kings.
#[rustfmt::skip]
const KK_IDX: [[u64; 64]; 10] = [[
     Z0,  Z0,  Z0,   0,   1,   2,   3,   4,
     Z0,  Z0,  Z0,   5,   6,   7,   8,   9,
     10,  11,  12,  13,  14,  15,  16,  17,
     18,  19,  20,  21,  22,  23,  24,  25,
     26,  27,  28,  29,  30,  31,  32,  33,
     34,  35,  36,  37,  38,  39,  40,  41,
     42,  43,  44,  45,  46,  47,  48,  49,
     50,  51,  52,  53,  54,  55,  56,  57,
], [
     58,  Z0,  Z0,  Z0,  59,  60,  61,  62,
     63,  Z0,  Z0,  Z0,  64,  65,  66,  67,
     68,  69,  70,  71,  72,  73,  74,  75,
     76,  77,  78,  79,  80,  81,  82,  83,
     84,  85,  86,  87,  88,  89,  90,  91,
     92,  93,  94,  95,  96,  97,  98,  99,
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
], [
    116, 117,  Z0,  Z0,  Z0, 118, 119, 120,
    121, 122,  Z0,  Z0,  Z0, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 171, 172, 173,
], [
    174,  Z0,  Z0,  Z0, 175, 176, 177, 178,
    179,  Z0,  Z0,  Z0, 180, 181, 182, 183,
    184,  Z0,  Z0,  Z0, 185, 186, 187, 188,
    189, 190, 191, 192, 193, 194, 195, 196,
    197, 198, 199, 200, 201, 202, 203, 204,
    205, 206, 207, 208, 209, 210, 211, 212,
    213, 214, 215, 216, 217, 218, 219, 220,
    221, 222, 223, 224, 225, 226, 227, 228,
], [
    229, 230,  Z0,  Z0,  Z0, 231, 232, 233,
    234, 235,  Z0,  Z0,  Z0, 236, 237, 238,
    239, 240,  Z0,  Z0,  Z0, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 255, 256, 257, 258, 259,
    260, 261, 262, 263, 264, 265, 266, 267,
    268, 269, 270, 271, 272, 273, 274, 275,
    276, 277, 278, 279, 280, 281, 282, 283,
], [
    284, 285, 286, 287, 288, 289, 290, 291,
    292, 293,  Z0,  Z0,  Z0, 294, 295, 296,
    297, 298,  Z0,  Z0,  Z0, 299, 300, 301,
    302, 303,  Z0,  Z0,  Z0, 304, 305, 306,
    307, 308, 309, 310, 311, 312, 313, 314,
    315, 316, 317, 318, 319, 320, 321, 322,
    323, 324, 325, 326, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338,
], [
     Z0,  Z0, 339, 340, 341, 342, 343, 344,
     Z0,  Z0, 345, 346, 347, 348, 349, 350,
     Z0,  Z0, 441, 351, 352, 353, 354, 355,
     Z0,  Z0,  Z0, 442, 356, 357, 358, 359,
     Z0,  Z0,  Z0,  Z0, 443, 360, 361, 362,
     Z0,  Z0,  Z0,  Z0,  Z0, 444, 363, 364,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 445, 365,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 446,
], [
     Z0,  Z0,  Z0, 366, 367, 368, 369, 370,
     Z0,  Z0,  Z0, 371, 372, 373, 374, 375,
     Z0,  Z0,  Z0, 376, 377, 378, 379, 380,
     Z0,  Z0,  Z0, 447, 381, 382, 383, 384,
     Z0,  Z0,  Z0,  Z0, 448, 385, 386, 387,
     Z0,  Z0,  Z0,  Z0,  Z0, 449, 388, 389,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 450, 390,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 451,
], [
    452, 391, 392, 393, 394, 395, 396, 397,
     Z0,  Z0,  Z0,  Z0, 398, 399, 400, 401,
     Z0,  Z0,  Z0,  Z0, 402, 403, 404, 405,
     Z0,  Z0,  Z0,  Z0, 406, 407, 408, 409,
     Z0,  Z0,  Z0,  Z0, 453, 410, 411, 412,
     Z0,  Z0,  Z0,  Z0,  Z0, 454, 413, 414,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 455, 415,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 456,
], [
    457, 416, 417, 418, 419, 420, 421, 422,
     Z0, 458, 423, 424, 425, 426, 427, 428,
     Z0,  Z0,  Z0,  Z0,  Z0, 429, 430, 431,
     Z0,  Z0,  Z0,  Z0,  Z0, 432, 433, 434,
     Z0,  Z0,  Z0,  Z0,  Z0, 435, 436, 437,
     Z0,  Z0,  Z0,  Z0,  Z0, 459, 438, 439,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 460, 440,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 461,
]];

/// Encoding of a pair of identical pieces.
#[rustfmt::skip]
const PP_IDX: [[u64; 64]; 10] = [[
      0,  Z0,   1,   2,   3,   4,   5,   6,
      7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,
     23,  24,  25,  26,  27,  28,  29,  30,
     31,  32,  33,  34,  35,  36,  37,  38,
     39,  40,  41,  42,  43,  44,  45,  46,
     Z0,  47,  48,  49,  50,  51,  52,  53,
     54,  55,  56,  57,  58,  59,  60,  61,
], [
     62,  Z0,  Z0,  63,  64,  65,  Z0,  66,
     Z0,  67,  68,  69,  70,  71,  72,  Z0,
     73,  74,  75,  76,  77,  78,  79,  80,
     81,  82,  83,  84,  85,  86,  87,  88,
     89,  90,  91,  92,  93,  94,  95,  96,
     Z0,  97,  98,  99, 100, 101, 102, 103,
     Z0, 104, 105, 106, 107, 108, 109,  Z0,
    110,  Z0, 111, 112, 113, 114,  Z0, 115,
], [
    116,  Z0,  Z0,  Z0, 117,  Z0,  Z0, 118,
     Z0, 119, 120, 121, 122, 123, 124,  Z0,
     Z0, 125, 126, 127, 128, 129, 130,  Z0,
    131, 132, 133, 134, 135, 136, 137, 138,
     Z0, 139, 140, 141, 142, 143, 144, 145,
     Z0, 146, 147, 148, 149, 150, 151,  Z0,
     Z0, 152, 153, 154, 155, 156, 157,  Z0,
    158,  Z0,  Z0, 159, 160,  Z0,  Z0, 161,
], [
    162,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 163,
     Z0, 164,  Z0, 165, 166, 167, 168,  Z0,
     Z0, 169, 170, 171, 172, 173, 174,  Z0,
     Z0, 175, 176, 177, 178, 179, 180,  Z0,
     Z0, 181, 182, 183, 184, 185, 186,  Z0,
     Z0,  Z0, 187, 188, 189, 190, 191,  Z0,
     Z0, 192, 193, 194, 195, 196, 197,  Z0,
    198,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 199,
], [
    200,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 201,
     Z0, 202,  Z0,  Z0, 203,  Z0, 204,  Z0,
     Z0,  Z0, 205, 206, 207, 208,  Z0,  Z0,
     Z0, 209, 210, 211, 212, 213, 214,  Z0,
     Z0,  Z0, 215, 216, 217, 218, 219,  Z0,
     Z0,  Z0, 220, 221, 222, 223,  Z0,  Z0,
     Z0, 224,  Z0, 225, 226,  Z0, 227,  Z0,
    228,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 229,
], [
    230,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 231,
     Z0, 232,  Z0,  Z0,  Z0,  Z0, 233,  Z0,
     Z0,  Z0, 234,  Z0, 235, 236,  Z0,  Z0,
     Z0,  Z0, 237, 238, 239, 240,  Z0,  Z0,
     Z0,  Z0,  Z0, 241, 242, 243,  Z0,  Z0,
     Z0,  Z0, 244, 245, 246, 247,  Z0,  Z0,
     Z0, 248,  Z0,  Z0,  Z0,  Z0, 249,  Z0,
    250,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 251,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 259,
     Z0, 252,  Z0,  Z0,  Z0,  Z0, 260,  Z0,
     Z0,  Z0, 253,  Z0,  Z0, 261,  Z0,  Z0,
     Z0,  Z0,  Z0, 254, 262,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 255,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 256,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 257,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 258,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 268,  Z0,
     Z0,  Z0, 263,  Z0,  Z0, 269,  Z0,  Z0,
     Z0,  Z0,  Z0, 264, 270,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 265,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 266,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0, 267,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 274,  Z0,  Z0,
     Z0,  Z0,  Z0, 271, 275,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 272,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0, 273,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
], [
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 277,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0, 276,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,
     Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0,  Z0
]];

/// The a7-a5-c5 triangle.
const TEST45: Bitboard = Bitboard::new(0x1_0307_0000_0000);

const fn binomial(mut n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k > n - k {
        return binomial(n, n - k);
    }
    let mut r = 1;
    let mut d = 1;
    while d <= k {
        r = r * n / d;
        n -= 1;
        d += 1;
    }
    r
}

struct Consts {
    mult_idx: [[u64; 10]; 5],
    mult_factor: [u64; 5],

    map_pawns: [u64; 64],
    lead_pawn_idx: [[u64; 64]; 6],
    lead_pawns_size: [[u64; 4]; 6],
}

const CONSTS: Consts = {
    let mut mult_idx = [[0; 10]; 5];
    let mut mult_factor = [0; 5];

    let mut i = 0;
    while i < 5 {
        let mut s = 0;
        let mut j = 0;
        while j < 10 {
            mult_idx[i][j] = s;
            s += if i == 0 {
                1
            } else {
                binomial(MULT_TWIST[INV_TRIANGLE[j]], i as u64)
            };
            j += 1;
        }
        mult_factor[i] = s;
        i += 1;
    }

    let mut available_squares = 48;

    let mut map_pawns = [0; 64];
    let mut lead_pawn_idx = [[0; 64]; 6];
    let mut lead_pawns_size = [[0; 4]; 6];

    let mut lead_pawns_cnt = 1;
    while lead_pawns_cnt <= 5 {
        let mut file = 0;
        while file < 4 {
            let mut idx = 0;
            let mut rank = 1;
            while rank < 7 {
                let sq = file + 8 * rank;
                if lead_pawns_cnt == 1 {
                    available_squares -= 1;
                    map_pawns[sq] = available_squares;
                    available_squares -= 1;
                    map_pawns[sq ^ 0x7] = available_squares; // flip horizontal
                }
                lead_pawn_idx[lead_pawns_cnt][sq] = idx;
                idx += binomial(map_pawns[sq], lead_pawns_cnt as u64 - 1);
                rank += 1;
            }
            lead_pawns_size[lead_pawns_cnt][file] = idx;
            file += 1;
        }
        lead_pawns_cnt += 1;
    }

    Consts {
        mult_idx,
        mult_factor,
        map_pawns,
        lead_pawn_idx,
        lead_pawns_size,
    }
};

/// Header nibble to piece.
fn nibble_to_piece(nibble: u8) -> Piece {
    match nibble {
        1 => Piece::WhitePawn,
        2 => Piece::WhiteKnight,
        3 => Piece::WhiteBishop,
        4 => Piece::WhiteRook,
        5 => Piece::WhiteQueen,
        6 => Piece::WhiteKing,
        9 => Piece::BlackPawn,
        10 => Piece::BlackKnight,
        11 => Piece::BlackBishop,
        12 => Piece::BlackRook,
        13 => Piece::BlackQueen,
        14 => Piece::BlackKing,
        _ => unsafe { unreachable_unchecked() },
    }
}

/// List of up to `MAX_PIECES` pieces.
type Pieces = ArrayVec<Piece, MAX_PIECES>;

/// Parse a piece list.
fn parse_pieces(raf: &RandomAccessFile, ptr: usize, count: usize, side: Color) -> Pieces {
    let mut pieces = Pieces::new();
    for p in raf.read(ptr..ptr + count) {
        pieces.push(nibble_to_piece(match side {
            Color::White => *p & 0xf,
            Color::Black => *p >> 4,
        }));
    }

    pieces
}

/// Group pieces that will be encoded together.
fn group_pieces(pieces: &Pieces, material: &Material) -> ArrayVec<usize, MAX_PIECES> {
    let mut result = ArrayVec::new();

    // For positions without pawns: If there are at least 3 unique pieces then 3
    // unique pieces will form the leading group. Otherwise the two kings will
    // form the leading group.
    let first_len = if material.has_pawns() {
        0
    } else if material.unique_pieces() >= 3 {
        3
    } else if material.unique_pieces() == 2 {
        2
    } else {
        material.min_like_man()
    };

    if first_len > 0 {
        result.push(first_len);
    }

    // The remaining identical pieces are grouped together.
    result.extend(pieces[first_len..].chunk_by(Piece::eq).map(<[_]>::len));

    result
}

/// Description of the encoding used for a piece configuration.
#[derive(Debug, Clone)]
struct GroupData {
    pieces: Pieces,
    material: Material,
    lens: ArrayVec<usize, MAX_PIECES>,
    factors: ArrayVec<u64, { MAX_PIECES + 1 }>,
}

impl GroupData {
    fn new(pieces: Pieces, material: Material, order: [u8; 2], file: usize) -> GroupData {
        debug_assert!(pieces.len() >= 2);

        // Compute group lengths.
        let lens = group_pieces(&pieces, &material);

        // Compute a factor for each group.
        let pp = material.left(Role::Pawn) > 0 && material.right(Role::Pawn) > 0;
        let mut factors = ArrayVec::from([0; MAX_PIECES + 1]);
        factors.truncate(lens.len() + 1);
        let mut free_squares = 64 - lens[0] - if pp { lens[1] } else { 0 };
        let mut next = if pp { 2 } else { 1 };
        let mut idx = 1;
        let mut k = 0;

        while next < lens.len() || k == order[0] || k == order[1] {
            if k == order[0] {
                // Leading pawns or pieces.
                factors[0] = idx;

                if material.has_pawns() {
                    idx *= CONSTS.lead_pawns_size[lens[0]][file];
                } else if material.unique_pieces() >= 3 {
                    idx *= 31_332;
                } else if material.unique_pieces() == 2 {
                    idx *= 462;
                } else if material.min_like_man() == 2 {
                    idx *= 278;
                } else {
                    idx *= CONSTS.mult_factor[material.min_like_man() - 1];
                }
            } else if k == order[1] {
                // Remaining pawns.
                factors[1] = idx;
                idx *= binomial(48 - lens[0] as u64, lens[1] as u64);
            } else {
                // Remaining pieces.
                factors[next] = idx;
                idx *= binomial(free_squares as u64, lens[next] as u64);
                free_squares -= lens[next];
                next += 1;
            }
            k += 1;
        }

        factors[lens.len()] = idx;

        GroupData {
            pieces,
            material,
            lens,
            factors,
        }
    }
}

/// Indexes into table of remapped DTZ values.
#[derive(Debug)]
enum DtzMap {
    /// Normal 8-bit DTZ map.
    Normal { map_ptr: usize, by_wdl: [u16; 4] },
    /// Wide 16-bit DTZ map for very long endgames.
    Wide { map_ptr: usize, by_wdl: [u16; 4] },
}

impl DtzMap {
    fn read(&self, raf: &RandomAccessFile, wdl: Wdl, res: u16) -> u16 {
        let wdl = match wdl {
            Wdl::Win => 0,
            Wdl::Loss => 1,
            Wdl::CursedWin => 2,
            Wdl::BlessedLoss => 3,
            Wdl::Draw => unsafe { unreachable_unchecked() },
        };

        match *self {
            DtzMap::Normal { map_ptr, by_wdl } => {
                let offset = map_ptr + by_wdl[wdl] as usize + res as usize;
                *raf.read(offset) as _
            }

            DtzMap::Wide { map_ptr, by_wdl } => {
                let offset = map_ptr + 2 * (by_wdl[wdl] as usize + res as usize);
                LE::read_u16(raf.read(offset..offset + 2))
            }
        }
    }
}

/// Huffman symbol.
#[derive(Debug, Clone)]
struct Symbol {
    lr: [u8; 3],
    len: u8,
}

impl Symbol {
    fn new() -> Symbol {
        Symbol { lr: [0; 3], len: 0 }
    }

    fn left(&self) -> u16 {
        (u16::from(self.lr[1] & 0xf) << 8) | u16::from(self.lr[0])
    }

    fn right(&self) -> u16 {
        (u16::from(self.lr[2]) << 4) | (u16::from(self.lr[1]) >> 4)
    }
}

/// Description of encoding and compression.
#[derive(Debug)]
struct PairsData {
    /// Encoding flags.
    flags: Flag,
    /// Piece configuration encoding info.
    groups: GroupData,

    /// Block size in bytes.
    block_size: u32,
    /// About every span values there is a sparse index entry.
    span: u32,
    /// Number of blocks in the table.
    blocks_num: u32,

    /// Minimum length in bits of the Huffman symbols.
    min_symlen: u8,
    /// Offset of the lowest symbols for each length.
    lowest_sym: u64,
    /// 64-bit padded lowest symbols for each length.
    base: Vec<u64>,
    /// Number of values represented by a given Huffman symbol.
    symbols: Vec<Symbol>,

    /// Offset of the sparse index.
    sparse_index: u64,
    /// Size of the sparse index.
    sparse_index_size: u32,

    /// Offset of the block length table.
    block_lengths: u64,
    /// Size of the block length table, padded to be bigger than `blocks_num`.
    block_length_size: u32,

    /// Start of compressed data.
    data: u64,

    /// DTZ mapping.
    dtz_map: Option<DtzMap>,
}

impl PairsData {
    fn parse<T: TableDescriptor>(
        raf: &RandomAccessFile,
        mut ptr: usize,
        groups: GroupData,
    ) -> (PairsData, usize) {
        let flags = Flag::new(*raf.read(ptr));

        if flags.contains(&Flag::SINGLE_VALUE) {
            let single_value = if T::METRIC == Metric::Wdl {
                *raf.read(ptr + 1)
            } else {
                0
            };

            let pairs = PairsData {
                flags,
                min_symlen: single_value,
                groups,
                base: Vec::new(),
                block_lengths: 0,
                block_length_size: 0,
                block_size: 0,
                blocks_num: 0,
                data: 0,
                lowest_sym: 0,
                span: 0,
                sparse_index: 0,
                sparse_index_size: 0,
                symbols: Vec::new(),
                dtz_map: None,
            };

            return (pairs, ptr + 2);
        }

        // Read header.
        let header = raf.read(ptr..ptr + 10);

        let tb_size = groups.factors[groups.lens.len()];
        let block_size = 1u32 << header[1] as u32;
        debug_assert!(block_size <= MAX_BLOCK_SIZE as u32);

        let span = 1u32 << header[2] as u32;
        let sparse_index_size = tb_size.div_ceil(span as u64) as u32;
        let padding = header[3];
        let blocks_num = LE::read_u32(&header[4..]);
        let block_length_size = blocks_num + padding as u32;

        let max_symlen = header[8];
        debug_assert!(max_symlen <= 32);
        let min_symlen = header[9];
        debug_assert!(min_symlen <= 32);

        debug_assert!(max_symlen >= min_symlen);
        let h = max_symlen as usize - min_symlen as usize + 1;

        let lowest_sym = ptr as u64 + 10;

        // Initialize base.
        let mut base = vec![0u64; h];
        for i in (0..h - 1).rev() {
            let ptr = lowest_sym as usize + i * 2;
            let buf = raf.read(ptr..ptr + 4);
            let b = base[i + 1] + LE::read_u16(&buf[..2]) as u64;
            base[i] = (b - LE::read_u16(&buf[2..]) as u64) / 2;
            debug_assert!(base[i] * 2 >= base[i + 1]);
        }

        for (i, base) in base.iter_mut().enumerate() {
            *base <<= 64 - (min_symlen as u32 + i as u32);
        }

        // Initialize symbols.
        ptr += 10 + h * 2;
        let sym = LE::read_u16(raf.read(ptr..ptr + 2));
        ptr += 2;
        let btree = ptr;
        let mut symbols = vec![Symbol::new(); sym as _];
        let mut visited = vec![false; symbols.len()];
        for s in 0..sym {
            read_symbols(raf, btree, &mut symbols, &mut visited, s, 16);
        }
        ptr += symbols.len() * 3 + (symbols.len() & 1);

        // Result.
        let pairs = PairsData {
            flags,
            groups,

            block_size,
            span,
            blocks_num,

            min_symlen,
            lowest_sym,
            base,
            symbols,

            sparse_index: 0, // to be initialized later
            sparse_index_size,

            block_lengths: 0, // to be initialized later
            block_length_size,

            data: 0, // to be initialized later

            dtz_map: None, // to be initialized later
        };

        (pairs, ptr)
    }
}

/// Build the symbol table.
fn read_symbols(
    raf: &RandomAccessFile,
    btree: usize,
    symbols: &mut [Symbol],
    visited: &mut [bool],
    sym: u16,
    depth: u8,
) {
    if *visited.get(sym as usize).assume() {
        return;
    }

    let mut symbol = Symbol::new();
    let offset = btree + 3 * sym as usize;
    let data = raf.read(offset..offset + symbol.lr.len());
    symbol.lr.copy_from_slice(data);

    if symbol.right() == 0xfff {
        symbol.len = 0;
    } else {
        // Guard against stack overflow.
        let depth = depth - 1;

        read_symbols(raf, btree, symbols, visited, symbol.left(), depth);
        read_symbols(raf, btree, symbols, visited, symbol.right(), depth);

        let l = symbols[symbol.left() as usize].len;
        let r = symbols[symbol.right() as usize].len;
        symbol.len = l + r + 1;
    }

    symbols[sym as usize] = symbol;
    visited[sym as usize] = true;
}

/// Description of encoding and compression for both sides of a table.
#[derive(Debug)]
struct FileData {
    sides: ArrayVec<PairsData, 2>,
}

/// Small readahead buffer for the block length table.
struct BlockLengthBuffer {
    buffer: [u8; 2 * BlockLengthBuffer::CACHED_BLOCKS as usize],
    first_block: Option<u32>,
}

/// Readahead direction.
enum Readahead {
    Forward,
    Backward,
}

impl BlockLengthBuffer {
    const CACHED_BLOCKS: u32 = 16;

    fn new() -> BlockLengthBuffer {
        BlockLengthBuffer {
            buffer: [0; 2 * BlockLengthBuffer::CACHED_BLOCKS as usize],
            first_block: None,
        }
    }

    fn fill_buffer(&mut self, raf: &RandomAccessFile, d: &PairsData, first_block: u32) -> u32 {
        let offset = d.block_lengths as usize + first_block as usize * 2;
        let data = raf.read(offset..offset + self.buffer.len());
        self.buffer.copy_from_slice(data);
        self.first_block = Some(first_block);
        first_block
    }

    fn read(
        &mut self,
        raf: &RandomAccessFile,
        d: &PairsData,
        block: u32,
        readahead: Readahead,
    ) -> u16 {
        let first_block = match self.first_block {
            Some(b) if (b..b + BlockLengthBuffer::CACHED_BLOCKS).contains(&block) => b,
            _ => match readahead {
                Readahead::Forward => self.fill_buffer(raf, d, block),
                Readahead::Backward => {
                    let block = block.saturating_sub(BlockLengthBuffer::CACHED_BLOCKS - 1);
                    self.fill_buffer(raf, d, block)
                }
            },
        };

        let index = (block - first_block) as usize * 2;
        LE::read_u16(&self.buffer[index..])
    }
}

/// A Syzygy table.
#[derive(Debug)]
pub struct Table<T: TableDescriptor> {
    descriptor: PhantomData<T>,
    raf: RandomAccessFile,
    num_unique_pieces: usize,
    min_like_man: usize,
    files: ArrayVec<FileData, 4>,
}

impl<T: TableDescriptor> Table<T> {
    /// Open a table, parse the header, the headers of the sub-tables and
    /// prepare meta data required for decompression.
    pub fn new(path: &Path, material: NormalizedMaterial) -> io::Result<Table<T>> {
        debug_assert!(material.count() <= MAX_PIECES);

        let raf = RandomAccessFile::new(path)?;
        if T::MAGIC != raf.read(0..T::MAGIC.len()) {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }

        // Read layout flags.
        let layout = Layout::new(*raf.read(4));
        let has_pawns = layout.contains(&Layout::HAS_PAWNS);

        debug_assert_eq!(has_pawns, material.has_pawns());
        debug_assert_ne!(layout.contains(&Layout::SPLIT), material.is_symmetric());

        // Read group data.
        let pp = material.left(Role::Pawn) > 0 && material.right(Role::Pawn) > 0;
        let num_files = if has_pawns { 4 } else { 1 };
        let num_sides = if T::METRIC == Metric::Wdl && !material.is_symmetric() {
            2
        } else {
            1
        };

        let mut ptr = 5;

        let files: ArrayVec<_, 4> = io::Result::from_iter((0..num_files).map(|file| {
            let order = [
                [
                    *raf.read(ptr) & 0xf,
                    if pp { *raf.read(ptr + 1) & 0xf } else { 0xf },
                ],
                [
                    *raf.read(ptr) >> 4,
                    if pp { *raf.read(ptr + 1) >> 4 } else { 0xf },
                ],
            ];

            ptr += 1 + pp as usize;

            let sides = ArrayVec::<_, 2>::from_iter(Color::iter().take(num_sides).map(|side| {
                let pieces = parse_pieces(&raf, ptr, material.count(), side);
                let key = Material::from_iter(pieces.clone());
                debug_assert!(key.normalize() == material);
                GroupData::new(pieces, key, order[side as usize], file)
            }));

            ptr += material.count();

            Ok(sides)
        }))?;

        ptr += ptr & 1;

        // Ensure reference pawn goes first.
        debug_assert_eq!((files[0][0].pieces[0].role() == Role::Pawn), has_pawns);

        // Setup pairs.
        let mut files = ArrayVec::<_, 4>::from_iter(files.into_iter().map(|file| FileData {
            sides: ArrayVec::from_iter(file.into_iter().map(|side| {
                let (pairs, next_ptr) = PairsData::parse::<T>(&raf, ptr, side);
                ptr = next_ptr;
                pairs
            })),
        }));

        // Setup DTZ map.
        if T::METRIC == Metric::Dtz {
            let map_ptr = ptr;
            for file in &mut files {
                if file.sides[0].flags.contains(&Flag::MAPPED) {
                    let mut by_wdl = [0; 4];
                    if file.sides[0].flags.contains(&Flag::WIDE_DTZ) {
                        for idx in &mut by_wdl {
                            *idx = ((ptr - map_ptr + 2) / 2) as u16;
                            ptr += LE::read_u16(raf.read(ptr..ptr + 2)) as usize * 2 + 2;
                        }
                        file.sides[0].dtz_map = Some(DtzMap::Wide { map_ptr, by_wdl });
                    } else {
                        for idx in &mut by_wdl {
                            *idx = (ptr - map_ptr + 1) as u16;
                            ptr += *raf.read(ptr) as usize + 1;
                        }
                        file.sides[0].dtz_map = Some(DtzMap::Normal { map_ptr, by_wdl });
                    }
                }
            }

            ptr += ptr & 1;
        }

        // Setup sparse index.
        for file in &mut files {
            for side in &mut file.sides {
                side.sparse_index = ptr as u64;
                ptr += side.sparse_index_size as usize * 6;
            }
        }

        for file in &mut files {
            for side in &mut file.sides {
                side.block_lengths = ptr as u64;
                ptr += side.block_length_size as usize * 2;
            }
        }

        for file in &mut files {
            for side in &mut file.sides {
                ptr = (ptr + 0x3f) & !0x3f; // 64 byte alignment
                side.data = ptr as u64;
                ptr += side.blocks_num as usize * side.block_size as usize;
            }
        }

        // Result.
        Ok(Table {
            descriptor: PhantomData,
            raf,
            num_unique_pieces: material.unique_pieces(),
            min_like_man: material.min_like_man(),
            files,
        })
    }

    /// Retrieves the value stored for `idx` by decompressing Huffman coded
    /// symbols stored in the corresponding block of the table.
    fn decompress_pairs(&self, d: &PairsData, idx: u64) -> u16 {
        // Special case: The table stores only a single value.
        if d.flags.contains(&Flag::SINGLE_VALUE) {
            return d.min_symlen as _;
        }

        // Use the sparse index to jump very close to the correct block.
        let (block, lit_idx) = self.read_sparse_index(d, idx);

        // Now move forwards/backwards to find the correct block.
        let (block, mut lit_idx) = self.read_block_lengths(d, block, lit_idx);

        // Read block (and 4 bytes to allow a final symbol refill) into memory.
        let offset = (d.data + (block as u64 * d.block_size as u64)) as usize;
        let len = offset + d.block_size as usize + 4;
        let mut cursor = self.raf.read(offset..len);

        // Find sym, the Huffman symbol that encodes the value for idx.
        let mut buf = cursor.read_u64::<BE>().assume();
        let mut buf_size = 64;

        let mut sym;

        loop {
            let mut len = 0;
            while buf < *d.base.get(len).assume() {
                len += 1;
            }

            sym = ((buf - d.base.get(len).assume()) >> (64 - len - d.min_symlen as usize)) as u16;
            let offset = d.lowest_sym as usize + 2 * len;
            sym += LE::read_u16(self.raf.read(offset..offset + 2));

            if lit_idx < d.symbols.get(sym as usize).assume().len as i64 + 1 {
                break;
            }

            lit_idx -= d.symbols.get(sym as usize).assume().len as i64 + 1;
            len += d.min_symlen as usize;
            buf <<= len;
            buf_size -= len;

            // Refill the buffer.
            if buf_size <= 32 {
                buf_size += 32;
                buf |= u64::from(cursor.read_u32::<BE>().assume()) << (64 - buf_size);
            }
        }

        // Decompress Huffman symbol.
        let mut symbol = d.symbols.get(sym as usize).assume();

        loop {
            if symbol.len == 0 {
                return symbol.left();
            }

            let left_symbol = d.symbols.get(symbol.left() as usize).assume();
            if lit_idx < left_symbol.len as i64 + 1 {
                symbol = left_symbol;
            } else {
                lit_idx -= left_symbol.len as i64 + 1;
                symbol = d.symbols.get(symbol.right() as usize).assume();
            }
        }
    }

    /// Given a position, determine the unique (modulo symmetries) index into
    /// the corresponding sub-table.
    fn encode(&self, pos: &Position, key: Material) -> Option<(&PairsData, u64)> {
        let material = self.files[0].sides[0].groups.material;
        debug_assert_eq!(key.normalize(), material.normalize());

        let symmetric_btm = material.is_symmetric() && pos.turn() == Color::Black;
        let black_stronger = key != material;
        let pov = Color::from(symmetric_btm || black_stronger);
        let bside: bool = pos.turn().perspective(pov).into();

        let mut used = Bitboard::empty();
        let mut sqs: ArrayVec<Square, MAX_PIECES> = ArrayVec::new();

        // For pawns there are sub-tables for each file (a, b, c, d) the
        // leading pawn can be placed on.
        let file = &self.files[if material.has_pawns() {
            let reference_pawn = self.files[0].sides[0].groups.pieces[0];
            debug_assert_eq!(reference_pawn.role(), Role::Pawn);
            let color = reference_pawn.color().perspective(pov);
            let lead_pawns = pos.pawns(color);

            used |= lead_pawns;
            sqs.extend(lead_pawns.into_iter().map(|sq| sq.perspective(pov)));

            // Ensure squares[0] is the maximum with regard to map_pawns.
            for i in 1..sqs.len() {
                if CONSTS.map_pawns[sqs[0] as usize] < CONSTS.map_pawns[sqs[i] as usize] {
                    sqs.swap(0, i);
                }
            }
            if sqs[0].file() >= File::E {
                sqs[0].mirror().file() as usize
            } else {
                sqs[0].file() as usize
            }
        } else {
            0
        }];

        // WDL tables have sub-tables for each side to move.
        let side = &file.sides[if bside { file.sides.len() - 1 } else { 0 }];

        // DTZ tables store only one side to move. It is possible that we have
        // to check the other side (by doing a 1-ply search).
        if T::METRIC == Metric::Dtz
            && side.flags.contains(&Flag::STM) != bside
            && (!material.is_symmetric() || material.has_pawns())
        {
            return None;
        }

        // The sub-table has been determined.
        //
        // So far squares has been initialized with the leading pawns.
        // Also add the other pieces.
        let lead_pawns_count = sqs.len();

        for piece in side.groups.pieces.iter().skip(lead_pawns_count) {
            let candidates = pos.by_piece(piece.perspective(pov)) & !used;
            let sq = candidates.into_iter().next().assume();
            sqs.push(sq.perspective(pov));
            used |= sq.bitboard();
        }

        debug_assert!(sqs.len() >= 2);

        // Now we can compute the index according to the piece positions.
        if sqs[0].file() >= File::E {
            for sq in &mut sqs {
                *sq = sq.mirror();
            }
        }

        let mut idx = if material.has_pawns() {
            let mut idx = CONSTS.lead_pawn_idx[lead_pawns_count][sqs[0] as usize];
            sqs[1..lead_pawns_count].sort_unstable_by_key(|sq| CONSTS.map_pawns[*sq as usize]);
            for (i, &sq) in sqs.iter().enumerate().take(lead_pawns_count).skip(1) {
                idx += binomial(CONSTS.map_pawns[sq as usize], i as u64);
            }

            idx
        } else {
            if sqs[0].rank() >= Rank::Fifth {
                for sq in &mut sqs {
                    *sq = sq.flip();
                }
            }

            for i in 0..side.groups.lens[0] {
                if sqs[i].file().transpose() == sqs[i].rank() {
                    continue;
                }

                if sqs[i].rank().transpose() > sqs[i].file() {
                    for sq in &mut sqs[i..] {
                        *sq = sq.transpose();
                    }
                }

                break;
            }

            if self.num_unique_pieces > 2 {
                let x = u64::from(sqs[1] > sqs[0]);
                let y = u64::from(sqs[2] > sqs[0]) + u64::from(sqs[2] > sqs[1]);

                if sqs[0].transpose() != sqs[0] {
                    let (b, c) = (sqs[1] as u64, sqs[2] as u64);
                    TRIANGLE[sqs[0] as usize] * 63 * 62 + (b - x) * 62 + (c - y)
                } else if sqs[1].transpose() != sqs[1] {
                    let (a, c) = (sqs[0].rank() as u64, sqs[2] as u64);
                    6 * 63 * 62 + a * 28 * 62 + LOWER[sqs[1] as usize] * 62 + (c - y)
                } else if sqs[2].transpose() != sqs[2] {
                    let (a, b) = (sqs[0].rank() as u64, sqs[1].rank() as u64);
                    6 * 63 * 62 + 4 * 28 * 62 + a * 7 * 28 + (b - x) * 28 + LOWER[sqs[2] as usize]
                } else {
                    let a = sqs[0].rank() as u64;
                    let b = sqs[1].rank() as u64;
                    let c = sqs[2].rank() as u64;
                    6 * 63 * 62 + 4 * 28 * 62 + 4 * 7 * 28 + a * 7 * 6 + (b - x) * 6 + (c - y)
                }
            } else if self.num_unique_pieces == 2 {
                KK_IDX[TRIANGLE[sqs[0] as usize] as usize][sqs[1] as usize]
            } else if self.min_like_man == 2 {
                if TRIANGLE[sqs[0] as usize] > TRIANGLE[sqs[1] as usize] {
                    sqs.swap(0, 1);
                }

                if sqs[0].file() >= File::E {
                    for sq in &mut sqs {
                        *sq = sq.mirror();
                    }
                }

                if sqs[0].rank() >= Rank::Fifth {
                    for sq in &mut sqs {
                        *sq = sq.flip();
                    }
                }

                if sqs[0].rank().transpose() > sqs[0].file()
                    || (sqs[0].transpose() == sqs[0] && sqs[1].rank().transpose() > sqs[1].file())
                {
                    for sq in &mut sqs {
                        *sq = sq.transpose();
                    }
                }

                if TEST45.contains(sqs[1]) && TRIANGLE[sqs[0] as usize] == TRIANGLE[sqs[1] as usize]
                {
                    sqs.swap(0, 1);
                    for sq in &mut sqs {
                        *sq = sq.flip().transpose();
                    }
                }

                PP_IDX[TRIANGLE[sqs[0] as usize] as usize][sqs[1] as usize]
            } else {
                for i in 1..side.groups.lens[0] {
                    if TRIANGLE[sqs[0] as usize] > TRIANGLE[sqs[i] as usize] {
                        sqs.swap(0, i);
                    }
                }

                if sqs[0].file() >= File::E {
                    for sq in &mut sqs {
                        *sq = sq.mirror();
                    }
                }

                if sqs[0].rank() >= Rank::Fifth {
                    for sq in &mut sqs {
                        *sq = sq.flip();
                    }
                }

                if sqs[0].rank().transpose() > sqs[0].file() {
                    for sq in &mut sqs {
                        *sq = sq.transpose();
                    }
                }

                for i in 1..side.groups.lens[0] {
                    for j in (i + 1)..side.groups.lens[0] {
                        if MULT_TWIST[sqs[i] as usize] > MULT_TWIST[sqs[j] as usize] {
                            sqs.swap(i, j);
                        }
                    }
                }

                let mut idx =
                    CONSTS.mult_idx[side.groups.lens[0] - 1][TRIANGLE[sqs[0] as usize] as usize];
                for i in 1..side.groups.lens[0] {
                    idx += binomial(MULT_TWIST[sqs[i] as usize], i as u64);
                }

                idx
            }
        };

        idx *= side.groups.factors[0];

        // Encode remaining pawns.
        let mut next = 1;
        let mut pawns_left = material.left(Role::Pawn) > 0 && material.right(Role::Pawn) > 0;
        let mut group_square = side.groups.lens[0];
        for lens in side.groups.lens.iter().copied().skip(1) {
            let (prev_squares, group_squares) = sqs.split_at_mut(group_square);
            let group_squares = &mut group_squares[..lens];
            group_squares.sort_unstable();

            let mut n = 0;
            for (i, &gsq) in group_squares.iter().enumerate().take(lens) {
                let adjust = prev_squares[..group_square]
                    .iter()
                    .filter(|sq| gsq > **sq)
                    .count() as u64;
                n += binomial(gsq as u64 - adjust - 8 * pawns_left as u64, i as u64 + 1);
            }

            pawns_left = false;
            idx += n * side.groups.factors[next];
            group_square += side.groups.lens[next];
            next += 1;
        }

        Some((side, idx))
    }

    fn read_sparse_index(&self, d: &PairsData, idx: u64) -> (u32, i64) {
        let main_idx = idx / d.span as u64;
        debug_assert!(main_idx <= u32::MAX as u64);

        let offset = d.sparse_index as usize + 6 * main_idx as usize;
        let sparse_index_entry = self.raf.read(offset..offset + 6);
        let block = LE::read_u32(&sparse_index_entry[..4]);
        let offset = LE::read_u16(&sparse_index_entry[4..]) as i64;

        let mut lit_idx = idx as i64 % d.span as i64 - d.span as i64 / 2;
        lit_idx += offset;

        (block, lit_idx)
    }

    fn read_block_lengths(&self, d: &PairsData, mut block: u32, mut lit_idx: i64) -> (u32, i64) {
        let mut buffer = BlockLengthBuffer::new();

        if lit_idx < 0 {
            // Backward scan.
            while lit_idx < 0 {
                block -= 1;
                lit_idx += buffer.read(&self.raf, d, block, Readahead::Backward) as i64 + 1;
            }
        } else {
            // Forward scan.
            loop {
                let block_length = buffer.read(&self.raf, d, block, Readahead::Forward) as i64 + 1;
                if lit_idx < block_length {
                    break;
                }
                lit_idx -= block_length;
                block += 1;
            }
        }

        (block, lit_idx)
    }
}

/// A WDL Table.
pub type WdlTable = Table<Wdl>;

impl WdlTable {
    pub fn probe(&self, pos: &Position, material: Material) -> Wdl {
        let (side, idx) = self.encode(pos, material).assume();
        match self.decompress_pairs(side, idx) {
            0 => Wdl::Loss,
            1 => Wdl::BlessedLoss,
            2 => Wdl::Draw,
            3 => Wdl::CursedWin,
            4 => Wdl::Win,
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

/// A DTZ Table.
pub type DtzTable = Table<Dtz>;

impl DtzTable {
    pub fn probe(&self, pos: &Position, material: Material, wdl: Wdl) -> Option<u16> {
        let (side, idx) = self.encode(pos, material)?;
        let decompressed = self.decompress_pairs(side, idx);

        let res = match &side.dtz_map {
            Some(map) => map.read(&self.raf, wdl, decompressed),
            None => decompressed,
        };

        let stores_plies = match wdl {
            Wdl::Win => side.flags.contains(&Flag::WIN_PLIES),
            Wdl::Loss => side.flags.contains(&Flag::LOSS_PLIES),
            Wdl::CursedWin | Wdl::BlessedLoss => false,
            Wdl::Draw => unsafe { unreachable_unchecked() },
        };

        Some(if stores_plies { res } else { 2 * res })
    }
}
