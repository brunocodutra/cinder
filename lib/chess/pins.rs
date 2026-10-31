use crate::chess::{Bitboard, Board, Furl, IdxSet, Placement, Rays, Square, Unfurl, Wordboard};
use crate::{simd::*, util::Num};
use bytemuck::zeroed;

/// [`Piece`]s of a [`Color`] that can reach each [`Square`] without breaking a pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pins {
    pins: Placement,
    squares: Bitboard,
    indices: IdxSet,
}

impl Pins {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(board: &Board, ksq: Square) -> Self {
        let turn = board.turn;
        let rays = ksq.rays();
        let furled = board.furl(rays);
        let nearest = furled.by_color(turn) & furled.visible() & Rays::pins();
        let beyond = furled.blend(nearest, zeroed()).visible() & Rays::pins();
        let pinners = beyond & furled.pinners() & furled.by_color(!turn);
        let pinned = nearest & pinners.flood_ranks();

        if !pinned.any() {
            return Pins {
                pins: zeroed(),
                squares: Bitboard::full(),
                indices: IdxSet::full(),
            };
        }

        let pins = furled.mask(pinned).extend().mask(beyond).unfurl(rays);
        let squares = pins.vacant().into();

        Pins {
            pins,
            squares,
            indices: !IdxSet::new(board.squares()[turn].to_simd().find(
                rays.compress(pinned.to_bitmask()).extract::<0, 16>(),
                pinned.count().cast(),
            )),
        }
    }

    /// Squares outside of the pin line.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn unpinned(&self) -> Bitboard {
        self.squares
    }

    /// A [`Wordboard`] with unpinned [`Threats`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask(&self) -> Wordboard {
        self.pins.to_idx_set() | u16x64::splat(*self.indices)
    }
}
