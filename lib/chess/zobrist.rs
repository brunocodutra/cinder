use crate::chess::{Castles, File, Piece, PieceTo, Square};
use crate::{simd::*, util::Key};
use bytemuck::{Pod, Zeroable, zeroed};

/// A type representing a [`Position`]'s [zobrist hashes](`Zobrists`).
pub type Zobrist = Key;

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
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
const ZOBRIST: Aligned<ZobristNumbers> = const {
    let mut zobrist: Aligned<ZobristNumbers> = zeroed();

    let numbers = zobrist.as_mut::<[u64; size_of::<ZobristNumbers>() / size_of::<u64>()]>();
    let mut state = 0x88C65730C3783F39u64;

    let mut i = 0;
    while i < numbers.len() {
        state = state.wrapping_add(0xA0761D6478BD642F);
        let hi_lo = Aligned((state as u128).wrapping_mul((state ^ 0xE7037ED1A0B428DB) as u128));
        let [hi, lo] = hi_lo.as_ref::<[u64; 2]>();
        numbers[i] = hi ^ lo;
        i += 1;
    }

    zobrist
};

const impl ZobristNumbers {
    #[inline(always)]
    pub fn psq(piece: Piece, sq: Square) -> Zobrist {
        Zobrist::new(ZOBRIST.pieces[piece][sq])
    }

    #[inline(always)]
    pub fn castling(castles: Castles) -> Zobrist {
        Zobrist::new(ZOBRIST.castles[castles])
    }

    #[inline(always)]
    pub fn en_passant(file: File) -> Zobrist {
        Zobrist::new(ZOBRIST.en_passant[file])
    }

    #[inline(always)]
    pub fn turn() -> Zobrist {
        Zobrist::new(ZOBRIST.turn)
    }
}
