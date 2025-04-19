use crate::chess::*;
use crate::util::{Bits, Integer};
use rand::prelude::*;
use std::{cell::SyncUnsafeCell, mem::MaybeUninit};

/// A type representing a [`Position`]'s [zobrist hash].
///
/// [zobrist hash]: https://www.chessprogramming.org/Zobrist_Hashing
pub type Zobrist = Bits<u64, 64>;

#[derive(Debug)]
pub struct ZobristNumbers {
    pieces: [[[u64; 64]; 6]; 2],
    castles: [u64; 16],
    en_passant: [u64; 8],
    turn: u64,
}

static ZOBRIST: SyncUnsafeCell<ZobristNumbers> = unsafe { MaybeUninit::zeroed().assume_init() };

#[cold]
#[ctor::ctor]
#[inline(never)]
unsafe fn init() {
    let zobrist = unsafe { ZOBRIST.get().as_mut_unchecked() };
    let mut rng = SmallRng::seed_from_u64(0x980E8CE238E3B114);
    zobrist.pieces = rng.random();
    zobrist.castles = rng.random();
    zobrist.en_passant = rng.random();
    zobrist.turn = rng.random();
}

impl ZobristNumbers {
    #[inline(always)]
    pub fn psq(color: Color, role: Role, sq: Square) -> Zobrist {
        let psq = unsafe { &ZOBRIST.get().as_ref_unchecked().pieces };
        Zobrist::new(psq[color as usize][role as usize][sq as usize])
    }

    #[inline(always)]
    pub fn castling(castles: Castles) -> Zobrist {
        let castling = unsafe { &ZOBRIST.get().as_ref_unchecked().castles };
        Zobrist::new(castling[castles.index() as usize])
    }

    #[inline(always)]
    pub fn en_passant(file: File) -> Zobrist {
        let en_passant = unsafe { &ZOBRIST.get().as_ref_unchecked().en_passant };
        Zobrist::new(en_passant[file as usize])
    }

    #[inline(always)]
    pub fn turn() -> Zobrist {
        Zobrist::new(unsafe { ZOBRIST.get().as_ref_unchecked().turn })
    }
}
