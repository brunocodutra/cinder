use crate::chess::*;
use crate::util::{Bits, Integer};
use bytemuck::{Zeroable, zeroed};
use rand::prelude::*;
use std::cell::SyncUnsafeCell;

/// A type representing a [`Position`]'s [zobrist hashes](`Zobrists`)
pub type Zobrist = Bits<u64, 64>;

#[derive(Debug, Zeroable)]
pub struct ZobristNumbers {
    pieces: PieceTo<u64>,
    castles: [u64; 16],
    en_passant: [u64; 8],
    turn: u64,
}

static ZOBRIST: SyncUnsafeCell<ZobristNumbers> = SyncUnsafeCell::new(zeroed());

#[cold]
#[ctor::ctor]
#[inline(never)]
unsafe fn init() {
    let zobrist = unsafe { ZOBRIST.get().as_mut_unchecked() };
    let mut rng = SmallRng::seed_from_u64(0x88C65730C3783F39);
    zobrist.pieces = rng.random();
    zobrist.castles = rng.random();
    zobrist.en_passant = rng.random();
    zobrist.turn = rng.random();
}

impl ZobristNumbers {
    #[inline(always)]
    pub fn psq(piece: Piece, sq: Square) -> Zobrist {
        let psq = unsafe { &ZOBRIST.get().as_ref_unchecked().pieces };
        Zobrist::new(psq[piece as usize][sq as usize])
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
