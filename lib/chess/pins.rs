use crate::chess::{Board, Furl, Threats, Unfurl, Wordboard};
use crate::util::Num;
use crate::{simd::*, util::Assume};
use bytemuck::zeroed;
use derive_more::with_trait::Deref;

/// [`Piece`]s of a [`Color`] that can reach each [`Square`] without breaking a pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deref)]
pub struct Pins {
    #[deref]
    attacks: Wordboard,
    unpinned: M8x64,
}

impl Pins {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(board: &Board, threats: &Threats) -> Self {
        let turn = board.turn;
        let ksq = board.king(turn).assume();
        let rays = ksq.rays();
        let pins = rays.pins();
        let furled = board.furl(rays);
        let ours = furled.by_color(turn);
        let theirs = furled.by_color(!turn);
        let nearest = ours & furled.visible() & pins;
        let beyond = furled.blend(nearest, zeroed()).visible() & pins;
        let pinners = beyond & furled.pinners() & theirs;
        let pinned = nearest & pinners.flood_ranks();

        if !pinned.any() {
            return Pins {
                attacks: threats[turn],
                unpinned: !zeroed::<M8x64>(),
            };
        }

        let piece_mask = board.squares()[turn].to_simd().find(
            rays.compress(pinned.to_bitmask()).extract::<0, 16>(),
            pinned.count().cast(),
        );

        let pinned = furled.mask(pinned).extend().mask(beyond).unfurl(rays);

        Pins {
            attacks: threats[turn] & (u16x64::splat(!piece_mask) | pinned.to_idx_set()),
            unpinned: !pinned.occupied(),
        }
    }

    /// A [`Wordboard`] with unpinned [`Threats`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn attacks(&self) -> &Wordboard {
        &self.attacks
    }

    /// Squares outside of the pin line.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn unpinned(&self) -> M8x64 {
        self.unpinned
    }
}
