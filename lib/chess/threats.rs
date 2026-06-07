use crate::util::{Assume, Int, Num};
use crate::{chess::*, simd::*};
use bytemuck::zeroed;
use derive_more::with_trait::{Debug, Deref, DerefMut};

/// [`Piece`]s threatening each [`Square`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
#[repr(transparent)]
pub struct Threats([Wordboard; Color::MAX as usize + 1]);

impl Threats {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(board: &Board) -> Self {
        let mut threats = Threats(zeroed());

        use Color::*;
        for sq in Square::iter() {
            let rays = sq.rays();
            let furled = board.furl(*rays);
            let attackers = furled.visible() & furled.attackers() & rays.valid();
            let black_attackers = attackers & furled.by_color(Black);
            let white_attackers = attackers & furled.by_color(White);

            threats[Black][sq] = IdxSet::new(board.squares()[Black].to_simd().find(
                rays.compress(*black_attackers).extract::<0, 16>(),
                black_attackers.len(),
            ));

            threats[White][sq] = IdxSet::new(board.squares()[White].to_simd().find(
                rays.compress(*white_attackers).extract::<0, 16>(),
                white_attackers.len(),
            ));
        }

        threats
    }

    /// Updates threats for a [`Place`] removed from a [`Square`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn outplace(&mut self, board: &Board, victim: Place, wt: Square) {
        let rays = wt.rays();
        let furled = board.furl(*rays);
        let pins = furled.visible() & rays.pins();
        let pinners = furled.mask(pins & furled.pinners());
        let pinners = pinners.extend().mask(pins.rotate_left(32));
        let pinners = pinners.unfurl(*rays.inv().flip());
        let indices = pinners.to_simd() & u8x64::splat(Place::IDX_MASK);
        let idx_set = u16x64::splat(1).shlv(indices.cast::<u16>());
        let updates = indices.simd_ne(zeroed()).select(idx_set, zeroed());
        let occ = pinners.by_color(Color::Black);

        match victim.color().assume() {
            Color::White => {
                self[Color::White] &= !u16x64::splat(victim.idx().assume().set().cast());
                self[Color::White] ^= occ.select(zeroed(), updates);
                self[Color::Black] ^= occ.select(updates, zeroed());
            }

            Color::Black => {
                self[Color::Black] &= !u16x64::splat(victim.idx().assume().set().cast());
                self[Color::Black] ^= occ.select(updates, zeroed());
                self[Color::White] ^= occ.select(zeroed(), updates);
            }
        }
    }

    /// Updates threats.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn replace(
        &mut self,
        board: &Board,
        src: Place,
        wc: Square,
        dst: Place,
        wt: Square,
        victim: Place,
    ) {
        let rays_wc = wc.rays();
        let rays_wt = wt.rays();

        let furled_wc = board.furl(*rays_wc);
        let furled_wt = board.splice(wc.bitboard(), zeroed()).furl(*rays_wt);

        let visible_wc = furled_wc.visible();
        let visible_wt = furled_wt.visible();

        let pins = visible_wc & rays_wc.pins();
        let pinners = furled_wc.mask(pins & furled_wc.pinners());
        let pinners = pinners.extend().mask(pins.rotate_left(32));
        let pinners = pinners.unfurl(*rays_wc.inv().flip());
        let indices = pinners.to_simd() & u8x64::splat(Place::IDX_MASK);
        let idx_set = u16x64::splat(1).shlv(indices.cast::<u16>());
        let updates = indices.simd_ne(zeroed()).select(idx_set, zeroed());

        let attacks = furled_wt.attacks(dst.piece().assume()) & visible_wt;
        let attacks = attacks.select(u8x64::splat(1), zeroed());
        let attacks = attacks.unfurl(*rays_wt.inv());
        let attacks = rays_wt.inv().valid().select(attacks, zeroed());

        let idx = src.idx().assume();
        let color = src.color().assume();
        debug_assert_eq!(dst.idx(), Some(idx));
        debug_assert_eq!(dst.color(), Some(color));
        debug_assert_eq!(victim.color(), Some(!color));

        let occ = pinners.by_color(Color::Black);

        match color {
            Color::White => {
                self[Color::White] &= !u16x64::splat(idx.set().cast());
                self[Color::White] |= attacks.cast::<u16>() << u16x64::splat(idx.cast());
                self[Color::White] ^= occ.select(zeroed(), updates);
                self[Color::Black] ^= occ.select(updates, zeroed());
                self[Color::Black] &= !u16x64::splat(victim.idx().assume().set().cast());
            }

            Color::Black => {
                self[Color::Black] &= !u16x64::splat(idx.set().cast());
                self[Color::Black] |= attacks.cast::<u16>() << u16x64::splat(idx.cast());
                self[Color::Black] ^= occ.select(updates, zeroed());
                self[Color::White] ^= occ.select(zeroed(), updates);
                self[Color::White] &= !u16x64::splat(victim.idx().assume().set().cast());
            }
        }
    }

    /// Updates threats.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn displace(&mut self, board: &Board, src: Place, wc: Square, dst: Place, wt: Square) {
        let rays_wc = wc.rays();
        let rays_wt = wt.rays();

        let furled_wc = board.furl(*rays_wc);
        let furled_wt = board.splice(wc.bitboard(), zeroed()).furl(*rays_wt);

        let visible_wc = furled_wc.visible();
        let visible_wt = furled_wt.visible();

        let pins_wc = visible_wc & rays_wc.pins();
        let pins_wt = visible_wt & rays_wt.pins();

        let pinners_wc = furled_wc.pinners();
        let pinners_wt = furled_wt.pinners();

        let pinners_wc = furled_wc.mask(pins_wc & pinners_wc);
        let pinners_wt = furled_wt.mask(pins_wt & pinners_wt);

        let pinners_wc = pinners_wc.extend();
        let pinners_wt = pinners_wt.extend();

        let pinners_wc = pinners_wc.mask(pins_wc.rotate_left(32));
        let pinners_wt = pinners_wt.mask(pins_wt.rotate_left(32));

        let pinners_wc = pinners_wc.unfurl(*rays_wc.inv().flip());
        let pinners_wt = pinners_wt.unfurl(*rays_wt.inv().flip());

        let indices_wc = pinners_wc.to_simd() & u8x64::splat(Place::IDX_MASK);
        let indices_wt = pinners_wt.to_simd() & u8x64::splat(Place::IDX_MASK);

        let idx_set_wc = u16x64::splat(1).shlv(indices_wc.cast::<u16>());
        let idx_set_wt = u16x64::splat(1).shlv(indices_wt.cast::<u16>());

        let updates_wc = indices_wc.simd_ne(zeroed()).select(idx_set_wc, zeroed());
        let updates_wt = indices_wt.simd_ne(zeroed()).select(idx_set_wt, zeroed());

        let attacks = furled_wt.attacks(dst.piece().assume()) & visible_wt;
        let attacks = attacks.select(u8x64::splat(1), zeroed());
        let attacks = attacks.unfurl(*rays_wt.inv());
        let attacks = rays_wt.inv().valid().select(attacks, zeroed());

        let idx = src.idx().assume();
        let color = src.color().assume();
        debug_assert_eq!(dst.idx(), Some(idx));
        debug_assert_eq!(dst.color(), Some(color));

        let occ_wc = pinners_wc.by_color(Color::Black);
        let occ_wt = pinners_wt.by_color(Color::Black);

        match color {
            Color::White => {
                self[Color::White] &= !u16x64::splat(idx.set().cast());
                self[Color::White] |= attacks.cast::<u16>() << u16x64::splat(idx.cast());
                self[Color::White] ^= occ_wc.select(zeroed(), updates_wc);
                self[Color::White] ^= occ_wt.select(zeroed(), updates_wt);
                self[Color::Black] ^= occ_wc.select(updates_wc, zeroed());
                self[Color::Black] ^= occ_wt.select(updates_wt, zeroed());
            }

            Color::Black => {
                self[Color::Black] &= !u16x64::splat(idx.set().cast());
                self[Color::Black] |= attacks.cast::<u16>() << u16x64::splat(idx.cast());
                self[Color::Black] ^= occ_wc.select(updates_wc, zeroed());
                self[Color::Black] ^= occ_wt.select(updates_wt, zeroed());
                self[Color::White] ^= occ_wc.select(zeroed(), updates_wc);
                self[Color::White] ^= occ_wt.select(zeroed(), updates_wt);
            }
        }
    }
}
