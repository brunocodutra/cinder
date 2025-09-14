use crate::nnue::Value;
use crate::util::{Binary, Bits, Bounded, Integer};
use crate::{chess::Flip, search::Ply};

/// Number of [plies][`Ply`] to mate.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum Mate {
    #[default]
    None,
    Mating(Ply),
    Mated(Ply),
}

impl Mate {
    #[inline(always)]
    pub fn plies(&self) -> Option<Ply> {
        match *self {
            Mate::None => None,
            Mate::Mating(ply) => Some(ply),
            Mate::Mated(ply) => Some(ply),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct ScoreRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

unsafe impl Integer for ScoreRepr {
    type Repr = i16;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 4095;
}

/// The minimax score.
pub type Score = Bounded<ScoreRepr>;

impl Score {
    const _CONDITION: () = const {
        assert!(Value::MAX + 2 * (Ply::MAX as i16 + 1) <= Self::MAX);
        assert!(Value::MIN + 2 * (Ply::MIN as i16 - 1) >= Self::MIN);
    };

    /// The tablebase loss score at `ply`.
    #[inline(always)]
    pub fn losing(ply: Ply) -> Self {
        Self::mated(Ply::upper()).relative_to_ply(ply) + 1
    }

    /// The maximum value.
    #[inline(always)]
    pub fn winning(ply: Ply) -> Self {
        Self::mating(Ply::upper()).relative_to_ply(ply) - 1
    }

    /// Mated score at `ply`
    #[inline(always)]
    pub fn mated(ply: Ply) -> Self {
        Self::lower().relative_to_ply(ply)
    }

    /// Mating score at `ply`
    #[inline(always)]
    pub fn mating(ply: Ply) -> Self {
        Self::upper().relative_to_ply(ply)
    }

    /// Returns number of plies to mate, if one is in the horizon.
    #[inline(always)]
    pub fn mate(&self) -> Mate {
        if self.is_loss() {
            Mate::Mated((*self - Score::lower()).saturate())
        } else if self.is_win() {
            Mate::Mating((Score::upper() - *self).saturate())
        } else {
            Mate::None
        }
    }

    /// Normalizes mate scores from `ply` relative to the root node.
    #[inline(always)]
    pub fn relative_to_root(&self, ply: Ply) -> Self {
        if self.is_decisive() {
            *self + ply.cast::<i16>() * self.signum()
        } else {
            *self
        }
    }

    /// Normalizes mate scores from the root node relative to `ply`.
    #[inline(always)]
    pub fn relative_to_ply(&self, ply: Ply) -> Self {
        if self.is_decisive() {
            *self - ply.cast::<i16>() * self.signum()
        } else {
            *self
        }
    }

    /// Returns true if the score represents a winning position.
    #[inline(always)]
    pub fn is_win(&self) -> bool {
        *self > Value::MAX
    }

    /// Returns true if the score represents a losing position.
    #[inline(always)]
    pub fn is_loss(&self) -> bool {
        *self < Value::MIN
    }

    /// Returns true if the score represents a decisive position (win or loss).
    #[inline(always)]
    pub fn is_decisive(&self) -> bool {
        self.is_win() || self.is_loss()
    }
}

impl Flip for Score {
    #[inline(always)]
    fn flip(self) -> Self {
        -self
    }
}

impl Binary for Score {
    type Bits = Bits<u16, 13>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        Bits::new((self.get() - Self::MIN + 1).cast())
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Score::new(bits.cast::<i16>() + Self::MIN - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn relative_to_root_ignores_non_mate_scores(
        #[filter(#s.mate() == Mate::None)] s: Score,
        p: Ply,
    ) {
        assert_eq!(s.relative_to_root(p), s);
    }

    #[proptest]
    fn relative_to_ply_ignores_non_mate_scores(
        #[filter(#s.mate() == Mate::None)] s: Score,
        p: Ply,
    ) {
        assert_eq!(s.relative_to_ply(p), s);
    }

    #[proptest]
    fn mate_returns_plies_to_mate(p: Ply) {
        assert_eq!(Score::mating(p).mate(), Mate::Mating(p));
    }

    #[proptest]
    fn mate_returns_plies_to_mated(p: Ply) {
        assert_eq!(Score::mated(p).mate(), Mate::Mated(p));
    }

    #[proptest]
    fn mating_implies_is_win(p: Ply) {
        assert!(Score::mating(p).is_win());
    }

    #[proptest]
    fn mated_implies_is_loss(p: Ply) {
        assert!(Score::mated(p).is_loss());
    }

    #[proptest]
    fn mating_implies_is_decisive(p: Ply) {
        assert!(Score::mating(p).is_decisive());
    }

    #[proptest]
    fn mated_implies_is_decisive(p: Ply) {
        assert!(Score::mated(p).is_decisive());
    }

    #[proptest]
    fn flipping_score_returns_its_negative(s: Score) {
        assert_eq!(s.flip(), -s);
    }

    #[proptest]
    fn decoding_encoded_score_is_an_identity(s: Score) {
        assert_eq!(Score::decode(s.encode()), s);
    }

    #[proptest]
    fn decoding_encoded_optional_score_is_an_identity(s: Option<Score>) {
        assert_eq!(Option::decode(s.encode()), s);
    }
}
