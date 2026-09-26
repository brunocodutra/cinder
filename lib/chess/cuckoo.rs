use crate::chess::{Move, Piece, Role, Square, Zobrist, ZobristNumbers};
use crate::util::{Assume, Int, Num, ones};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;
use std::ops::Shr;

#[derive(Debug, Zeroable)]
pub struct Cuckoo {
    keys: [Zobrist; Cuckoo::LEN],
    moves: [Option<Move>; Cuckoo::LEN],
}

const impl Cuckoo {
    const LEN: usize = 1 << 13;
    const MASK: usize = ones(Self::LEN.trailing_zeros());

    #[inline(always)]
    fn h1(hash: Zobrist) -> usize {
        Self::MASK & hash.cast::<usize>().shr(32)
    }

    #[inline(always)]
    fn h2(hash: Zobrist) -> usize {
        Self::MASK & hash.cast::<usize>().shr(48)
    }

    #[inline(always)]
    pub fn find(hash: Zobrist) -> Option<Move> {
        let mut slot = Self::h1(hash);
        if *CUCKOO.keys.get(slot).assume() != hash {
            slot = Self::h2(hash);
            if *CUCKOO.keys.get(slot).assume() != hash {
                return None;
            }
        }

        *CUCKOO.moves.get(slot).assume()
    }
}

static CUCKOO: Cuckoo = const {
    // Null turn hash would make cuckoo colorblind, so no odd distance would ever match
    assert!(ZobristNumbers::turn() != zeroed());

    let mut cuckoo: Cuckoo = zeroed();

    for piece in Piece::iter() {
        if piece.role() != Role::Pawn {
            for wc in Square::iter() {
                let attacks = piece.attacks(wc);
                for wt in Square::iter() {
                    if wt.get() > wc.get() && attacks.contains(wt) {
                        let mut candidate = Some(Move::regular(wc, wt, None));
                        let mut key = ZobristNumbers::psq(piece, wc)
                            ^ ZobristNumbers::psq(piece, wt)
                            ^ ZobristNumbers::turn();

                        let mut steps: usize = 0;
                        let mut slot = Cuckoo::h1(key);
                        while cuckoo.moves[slot].is_some() {
                            steps += 1;
                            assert!(steps <= Cuckoo::LEN);

                            let victim = (cuckoo.moves[slot], cuckoo.keys[slot]);
                            (cuckoo.moves[slot], cuckoo.keys[slot]) = (candidate, key);
                            (candidate, key) = victim;

                            slot = if slot == Cuckoo::h1(key) {
                                Cuckoo::h2(key)
                            } else {
                                Cuckoo::h1(key)
                            };
                        }

                        cuckoo.moves[slot] = candidate;
                        cuckoo.keys[slot] = key;
                    }
                }
            }
        }
    }

    let mut occupied = 0;
    let mut slot = Cuckoo::LEN;
    while slot > 0 {
        slot -= 1;
        if cuckoo.moves[slot].is_some() {
            occupied += 1;
        }
    }

    assert!(occupied == 3668);
    cuckoo
};
