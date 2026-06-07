use crate::chess::{Bitboard, Board, Furl, Threats, Unfurl, Wordboard};
use crate::{simd::*, util::Assume};
use bytemuck::zeroed;
use derive_more::with_trait::Deref;

/// [`Piece`]s of a [`Color`] that can reach each [`Square`] without breaking a pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deref)]
pub struct Pins {
    #[deref]
    attacks: Wordboard,
    mask: Bitboard,
}

impl Pins {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(board: &Board, threats: &Threats) -> Self {
        let turn = board.turn;
        let ksq = board.king(turn).assume();
        let rays = ksq.rays();
        let pins = rays.pins();
        let furled = board.furl(*rays);
        let ours = furled.by_color(turn);
        let theirs = furled.by_color(!turn);
        let nearest = ours & furled.visible() & pins;
        let beyond = furled.beyond(nearest) & pins;
        let pinners = beyond & furled.pinners() & theirs;
        let pinned = nearest & pinners.flood_ranks();

        if pinned == zeroed() {
            return Pins {
                mask: zeroed(),
                attacks: threats[turn],
            };
        }

        let squares = board.squares()[turn].to_simd();
        let piece_mask = squares.find(rays.compress(*pinned).extract::<0, 16>(), pinned.len());

        let pinned = furled.mask(pinned).extend();
        let pinned = pinned.mask(beyond).unfurl(*rays.inv());
        let mask = pinned.to_simd().simd_ne(zeroed()).into();
        let idx_set = u16x64::splat(1).shlv(pinned.indices().cast::<u16>());

        Pins {
            mask,
            attacks: threats[turn] & (u16x64::splat(!piece_mask) | mask.select(idx_set, zeroed())),
        }
    }

    /// A [`Wordboard`] with unpinned [`Threats`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attacks(&self) -> &Wordboard {
        &self.attacks
    }

    /// A [`Bitboard`] for pins.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn mask(&self) -> Bitboard {
        self.mask
    }
}
