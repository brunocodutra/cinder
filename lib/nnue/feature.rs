use crate::chess::{Color, File, Perspective, Piece, Role, Side, Square};
use crate::util::{Assume, Int, Num, ones};
use std::ops::{BitAnd, Index, IndexMut};

/// A PSQ feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PSQFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PSQFeature as Num>::Repr);

const unsafe impl Num for PSQFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for PSQFeature {}

const impl PSQFeature {
    /// The total number of different features.
    pub const LEN: usize = 768;

    /// Constructs a [`PSQFeature`].
    #[inline(always)]
    pub fn new(side: Color, ksq: Square, piece: Piece, sq: Square) -> Self {
        let chirality = Side::from(ksq.file() < File::E);
        let psq = 64 * piece.perspective(side).cast::<u16>()
            + sq.perspective(side).perspective(chirality).cast::<u16>();

        Num::new(psq)
    }
}

const impl<T> Index<PSQFeature> for [T; PSQFeature::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: PSQFeature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<PSQFeature> for [T; PSQFeature::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, f: PSQFeature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}

/// The king's bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct KingBucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <KingBucket as Num>::Repr);

const unsafe impl Num for KingBucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 31;
}

const unsafe impl Int for KingBucket {}

const impl KingBucket {
    pub const LEN: usize = Self::MAX as usize + 1;

    #[inline(always)]
    pub fn new(side: Color, ksq: Square) -> Self {
        #[rustfmt::skip]
        const BUCKETS: [u8; 64] = [
            16, 17, 18, 19,  3,  2,  1,  0,
            20, 21, 22, 23,  7,  6,  5,  4,
            24, 25, 26, 27, 11, 10,  9,  8,
            24, 25, 26, 27, 11, 10,  9,  8,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
        ];

        Num::new(BUCKETS[ksq.perspective(side)])
    }
}

const impl<T> Index<KingBucket> for [T; KingBucket::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, b: KingBucket) -> &Self::Output {
        self.get(b.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<KingBucket> for [T; KingBucket::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, b: KingBucket) -> &mut Self::Output {
        self.get_mut(b.cast::<usize>()).assume()
    }
}

/// A king-bucketed PSQ feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct KAFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <KAFeature as Num>::Repr);

const unsafe impl Num for KAFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for KAFeature {}

const impl KAFeature {
    /// The total number of different features.
    pub const LEN: usize = PSQFeature::LEN * KingBucket::LEN / 2;

    /// Constructs a [`KAFeature`].
    #[inline(always)]
    pub fn new(side: Color, ksq: Square, piece: Piece, sq: Square) -> Self {
        let chirality = Side::from(ksq.file() < File::E);
        let psq = PSQFeature::new(side, ksq, piece, sq).get();
        let bucket = KingBucket::new(side, ksq.perspective(chirality));
        Num::new(psq + PSQFeature::LEN.cast::<u16>() * bucket.cast::<u16>())
    }
}

const impl<T> Index<KAFeature> for [T; KAFeature::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: KAFeature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<KAFeature> for [T; KAFeature::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, f: KAFeature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}

/// A threat feature with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct TIFeature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <TIFeature as Num>::Repr);

const unsafe impl Num for TIFeature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for TIFeature {}

const fn threat_index_for(role: Role) -> [u16; 65] {
    let mut table = [0u16; 65];

    let mut count = 0;
    for sq in Square::iter() {
        table[sq as usize] = count;
        count += Piece::new(role, Color::White).attacks(sq).len() as u16;
    }

    table[64] = count;
    table
}

impl TIFeature {
    const PAWN_INDEX: u16 = 84;
    const KNIGHT_INDICES: [u16; 65] = threat_index_for(Role::Knight);
    const BISHOP_INDICES: [u16; 65] = threat_index_for(Role::Bishop);
    const ROOK_INDICES: [u16; 65] = threat_index_for(Role::Rook);
    const QUEEN_INDICES: [u16; 65] = threat_index_for(Role::Queen);

    const KNIGHT_OFFSET: u16 = 6 * Self::PAWN_INDEX;
    const BISHOP_OFFSET: u16 = Self::KNIGHT_OFFSET + 10 * Self::KNIGHT_INDICES[64];
    const ROOK_OFFSET: u16 = Self::BISHOP_OFFSET + 8 * Self::BISHOP_INDICES[64];
    const QUEEN_OFFSET: u16 = Self::ROOK_OFFSET + 8 * Self::ROOK_INDICES[64];
    const THREAT_FEATURES: u16 = Self::QUEEN_OFFSET + 10 * Self::QUEEN_INDICES[64];

    /// The total number of different features.
    pub const LEN: usize = 2 * Self::THREAT_FEATURES as usize;

    /// Constructs a [`ThreatFeature`].
    #[inline(always)]
    pub fn new(
        side: Color,
        ksq: Square,
        src: Piece,
        wc: Square,
        dst: Piece,
        wt: Square,
    ) -> Option<Self> {
        if src.role() == Role::King || dst.role() == Role::King {
            return None;
        }

        let chirality = Side::from(ksq.file() < File::E);
        let wc = wc.perspective(side).perspective(chirality);
        let wt = wt.perspective(side).perspective(chirality);
        let src = src.perspective(side);
        let dst = dst.perspective(side);

        let idx = match src.role() {
            Role::Pawn => Self::pawn_threat_idx(src, wc, dst, wt),
            Role::Knight => Self::knight_threat_idx(src, wc, dst, wt),
            Role::Bishop => Self::bishop_threat_idx(src, wc, dst, wt),
            Role::Rook => Self::rook_threat_idx(src, wc, dst, wt),
            Role::Queen => Self::queen_threat_idx(src, wc, dst, wt),
            Role::King => None,
        }?;

        Some(Num::new(
            idx + src.color().cast::<u16>() * Self::THREAT_FEATURES,
        ))
    }

    #[inline(always)]
    fn pawn_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        if wt > wc && src.role() == dst.role() && src.color() != dst.color() {
            return None;
        }

        let stride = Self::stride(src, dst)?;
        let rank = wc.rank().cast::<u16>() - 1;
        let file = wc.file().cast::<u16>();
        let leftward = wt.file() < wc.file();
        let attack = 2 * file - u16::from(leftward);
        Some(stride * Self::PAWN_INDEX + rank * 14 + attack)
    }

    #[inline(always)]
    fn knight_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::KNIGHT_INDICES, Self::KNIGHT_OFFSET)
    }

    #[inline(always)]
    fn bishop_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::BISHOP_INDICES, Self::BISHOP_OFFSET)
    }

    #[inline(always)]
    fn rook_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::ROOK_INDICES, Self::ROOK_OFFSET)
    }

    #[inline(always)]
    fn queen_threat_idx(src: Piece, wc: Square, dst: Piece, wt: Square) -> Option<u16> {
        Self::piece_threat_idx(src, wc, dst, wt, Self::QUEEN_INDICES, Self::QUEEN_OFFSET)
    }

    #[inline(always)]
    fn piece_threat_idx(
        src: Piece,
        wc: Square,
        dst: Piece,
        wt: Square,
        indices: [u16; 65],
        offset: u16,
    ) -> Option<u16> {
        if wt > wc && src.role() == dst.role() {
            return None;
        }

        let stride = Self::stride(src, dst)?;
        let below = src.attacks(wc).bitand(ones::<u64>(wt.cast())).len();
        Some(offset + stride * indices[64] + indices[wc.cast::<usize>()] + below.cast::<u16>())
    }

    #[inline(always)]
    fn stride(src: Piece, dst: Piece) -> Option<u16> {
        const P: [[Option<u16>; 6]; 2] = [
            [Some(0), Some(1), None, Some(2), None, None],
            [Some(3), Some(4), None, Some(5), None, None],
        ];

        const KQ: [[Option<u16>; 6]; 2] = [
            [Some(0), Some(1), Some(2), Some(3), Some(4), None],
            [Some(5), Some(6), Some(7), Some(8), Some(9), None],
        ];

        const BR: [[Option<u16>; 6]; 2] = [
            [Some(0), Some(1), Some(2), Some(3), None, None],
            [Some(4), Some(5), Some(6), Some(7), None, None],
        ];

        match src.role() {
            Role::Pawn => P[dst.color()][dst.role()],
            Role::Knight | Role::Queen => KQ[dst.color()][dst.role()],
            Role::Bishop | Role::Rook => BR[dst.color()][dst.role()],
            Role::King => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Flip, Mirror, Move, Position};
    use proptest::{prop_assume, sample::Selector};
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ka_feature_is_unique_to_perspective(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_ne!(
            KAFeature::new(c, ksq, p, sq),
            KAFeature::new(!c, ksq, p, sq)
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn threat_feature_is_unique_to_perspective(
        #[filter(#pos.noisy().into_iter().any(|m| !#pos[m.whither()].is_empty()))] pos: Position,
        #[map(|s: Selector| s.select(#pos.noisy().into_iter().filter(|m| !#pos[m.whither()].is_empty())))]
        m: Move,
    ) {
        let c = pos.turn();
        let ksq = pos.king(c);
        let (wc, wt) = (m.whence(), m.whither());
        let (src, dst) = (pos[wc].piece().unwrap(), pos[wt].piece().unwrap());

        let feature = TIFeature::new(c, ksq, src, wc, dst, wt);
        prop_assume!(feature.is_some());

        assert_ne!(feature, TIFeature::new(!c, ksq, src, wc, dst, wt));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ka_feature_is_vertically_symmetric(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_eq!(
            KAFeature::new(c, ksq, p, sq),
            KAFeature::new(c.flip(), ksq.flip(), p.flip(), sq.flip())
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn threat_feature_is_vertically_symmetric(
        #[filter(#pos.noisy().into_iter().any(|m| !#pos[m.whither()].is_empty()))] pos: Position,
        #[map(|s: Selector| s.select(#pos.noisy().into_iter().filter(|m| !#pos[m.whither()].is_empty())))]
        m: Move,
    ) {
        let c = pos.turn();
        let ksq = pos.king(c);
        let (wc, wt) = (m.whence(), m.whither());
        let (src, dst) = (pos[wc].piece().unwrap(), pos[wt].piece().unwrap());

        assert_eq!(
            TIFeature::new(c, ksq, src, wc, dst, wt),
            TIFeature::new(
                c.flip(),
                ksq.flip(),
                src.flip(),
                wc.flip(),
                dst.flip(),
                wt.flip()
            )
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn ka_feature_is_horizontally_symmetric(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_eq!(
            KAFeature::new(c, ksq, p, sq),
            KAFeature::new(c, ksq.mirror(), p, sq.mirror())
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn threat_feature_is_horizontally_symmetric(
        #[filter(#pos.noisy().into_iter().any(|m| !#pos[m.whither()].is_empty()))] pos: Position,
        #[map(|s: Selector| s.select(#pos.noisy().into_iter().filter(|m| !#pos[m.whither()].is_empty())))]
        m: Move,
    ) {
        let c = pos.turn();
        let ksq = pos.king(c);
        let (wc, wt) = (m.whence(), m.whither());
        let (src, dst) = (pos[wc].piece().unwrap(), pos[wt].piece().unwrap());

        assert_eq!(
            TIFeature::new(c, ksq, src, wc, dst, wt),
            TIFeature::new(c, ksq.mirror(), src, wc.mirror(), dst, wt.mirror())
        );
    }
}
