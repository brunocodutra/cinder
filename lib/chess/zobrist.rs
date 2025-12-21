use crate::{chess::*, util::Key};
use bytemuck::{Pod, Zeroable, zeroed};
use std::mem::transmute;

/// A type representing a [`Position`]'s [zobrist hashes](`Zobrists`)
pub type Zobrist = Key;

#[derive(Debug, Copy, Clone, Zeroable, Pod)]
#[repr(C)]
pub struct ZobristNumbers {
    pieces: PieceTo<u64>,
    castles: [u64; 16],
    en_passant: [u64; 8],
    turn: u64,
}

/// Zobrist constants initialized using the [Wyrand] PRNG.
///
/// [Wyrand]: https://github.com/wangyi-fudan/wyhash
static ZOBRIST: ZobristNumbers = const {
    let mut zobrist: ZobristNumbers = zeroed();

    const NUMBERS: usize = size_of::<ZobristNumbers>() / size_of::<u64>();
    let numbers = unsafe { transmute::<&mut ZobristNumbers, &mut [u64; NUMBERS]>(&mut zobrist) };
    let mut state = 0x88C65730C3783F39u64;

    let mut i = 0;
    while i < numbers.len() {
        state = state.wrapping_add(0xA0761D6478BD642F);
        let hi_lo = (state as u128).wrapping_mul((state ^ 0xE7037ED1A0B428DB) as u128);
        let [hi, lo] = unsafe { transmute::<u128, [u64; 2]>(hi_lo) };
        numbers[i] = hi ^ lo;
        i += 1;
    }

    zobrist
};

impl ZobristNumbers {
    #[inline(always)]
    pub const fn psq(piece: Piece, sq: Square) -> Zobrist {
        Zobrist::new(ZOBRIST.pieces[piece as usize][sq as usize])
    }

    #[inline(always)]
    pub const fn castling(castles: Castles) -> Zobrist {
        Zobrist::new(ZOBRIST.castles[castles.index() as usize])
    }

    #[inline(always)]
    pub const fn en_passant(file: File) -> Zobrist {
        Zobrist::new(ZOBRIST.en_passant[file as usize])
    }

    #[inline(always)]
    pub const fn turn() -> Zobrist {
        Zobrist::new(ZOBRIST.turn)
    }
}
