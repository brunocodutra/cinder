use crate::chess::{Flip, Halfmove};
use crate::search::{Ply, Value};
use crate::util::{Binary, Bits, Bounded, Int, Num};
use bytemuck::{NoUninit, Zeroable, zeroed};

/// The distance in plies to a decisive score.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum Decision {
    #[default]
    None,
    Winning(Ply),
    Losing(Ply),
    Mating(Ply),
    Mated(Ply),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct ScoreRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <ScoreRepr as Num>::Repr);

const unsafe impl Num for ScoreRepr {
    type Repr = i16;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 8191;
}

const unsafe impl Int for ScoreRepr {}

/// The minimax score.
pub type Score = Bounded<ScoreRepr>;

impl Score {
    /// The drawn score.
    #[inline(always)]
    pub fn drawn() -> Self {
        Self::new(0)
    }

    /// The tablebase losing score at `ply`.
    #[inline(always)]
    pub fn losing(ply: Ply) -> Self {
        Self::mated(Ply::upper()).relative_to_ply(ply) + 1
    }

    /// The tablebase winning score at `ply`.
    #[inline(always)]
    pub fn winning(ply: Ply) -> Self {
        Self::mating(Ply::upper()).relative_to_ply(ply) - 1
    }

    /// Mated score at `ply`.
    #[inline(always)]
    pub fn mated(ply: Ply) -> Self {
        Self::lower().relative_to_ply(ply)
    }

    /// Mating score at `ply`.
    #[inline(always)]
    pub fn mating(ply: Ply) -> Self {
        Self::upper().relative_to_ply(ply)
    }

    /// Returns the distance in plies to a decisive score, if one is in the horizon.
    #[inline(always)]
    pub fn decision(self) -> Decision {
        if self.is_mating() {
            Decision::Mating(Ply::new(Score::MAX - self.get()))
        } else if self.is_mated() {
            Decision::Mated(Ply::new(self.get() - Score::MIN))
        } else if self.is_winning() {
            Decision::Winning(Ply::upper() - (self.get() - Value::MAX - 1))
        } else if self.is_losing() {
            Decision::Losing(Ply::upper() - (Value::MIN - self.get() - 1))
        } else {
            Decision::None
        }
    }

    /// Lowers scores that cannot be reached before the 50-move rule.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn downgrade(self, hm: Halfmove) -> Self {
        match self.decision() {
            Decision::Mating(p) | Decision::Winning(p) if p > Halfmove::upper() - hm => {
                Self::new(Value::MAX)
            }

            Decision::Mated(p) | Decision::Losing(p) if p > Halfmove::upper() - hm => {
                Self::new(Value::MIN)
            }

            _ => self,
        }
    }

    /// Normalizes mate scores from `ply` relative to the root node.
    #[inline(always)]
    pub fn relative_to_root(self, ply: Ply) -> Self {
        if self.is_winning() {
            self + ply
        } else if self.is_losing() {
            self - ply
        } else {
            self
        }
    }

    /// Normalizes mate scores from the root node relative to `ply`.
    #[inline(always)]
    pub fn relative_to_ply(self, ply: Ply) -> Self {
        if self.is_winning() {
            self - ply
        } else if self.is_losing() {
            self + ply
        } else {
            self
        }
    }

    /// Returns true if the score represents a winning position.
    #[inline(always)]
    pub fn is_winning(self) -> bool {
        self > Value::MAX
    }

    /// Returns true if the score represents a losing position.
    #[inline(always)]
    pub fn is_losing(self) -> bool {
        self < Value::MIN
    }

    /// Returns true if the score represents a winning or losing position.
    #[inline(always)]
    pub fn is_decisive(self) -> bool {
        self.is_winning() || self.is_losing()
    }

    /// Returns true if the score represents a won position.
    #[inline(always)]
    pub fn is_mating(self) -> bool {
        self > Self::winning(zeroed())
    }

    /// Returns true if the score represents a lost position.
    #[inline(always)]
    pub fn is_mated(self) -> bool {
        self < Self::losing(zeroed())
    }
}

impl Flip for Score {
    #[inline(always)]
    fn flip(self) -> Self {
        -self
    }
}

impl Binary for Score {
    type Bits = Bits<u16, 14>;

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
    fn relative_to_root_ignores_undecided_scores(
        #[filter(!#s.is_winning() && !#s.is_losing())] s: Score,
        p: Ply,
    ) {
        assert_eq!(s.relative_to_root(p), s);
    }

    #[proptest]
    fn relative_to_ply_ignores_undecided_scores(
        #[filter(!#s.is_winning() && !#s.is_losing())] s: Score,
        p: Ply,
    ) {
        assert_eq!(s.relative_to_ply(p), s);
    }

    #[proptest]
    fn decision_returns_plies_to_mate(p: Ply) {
        assert_eq!(Score::mating(p).decision(), Decision::Mating(p));
    }

    #[proptest]
    fn decision_returns_plies_to_mated(p: Ply) {
        assert_eq!(Score::mated(p).decision(), Decision::Mated(p));
    }

    #[proptest]
    fn decision_returns_plies_to_tb_win(p: Ply) {
        assert_eq!(Score::winning(p).decision(), Decision::Winning(p));
    }

    #[proptest]
    fn decision_returns_plies_to_tb_loss(p: Ply) {
        assert_eq!(Score::losing(p).decision(), Decision::Losing(p));
    }

    #[proptest]
    fn mating_implies_is_mating(p: Ply) {
        assert!(Score::mating(p).is_mating());
    }

    #[proptest]
    fn mated_implies_is_mated(p: Ply) {
        assert!(Score::mated(p).is_mated());
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
