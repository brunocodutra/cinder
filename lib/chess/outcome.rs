/// One of the possible outcomes of a chess game.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum Outcome {
    Checkmate,
    Stalemate,
    DrawBy50MoveRule,
    DrawByThreefoldRepetition,
}

const impl Outcome {
    /// Whether the outcome is a draw and neither side has won.
    #[inline(always)]
    pub fn is_draw(self) -> bool {
        !self.is_decisive()
    }

    /// Whether the outcome is a decisive and one of the sides has won.
    #[inline(always)]
    pub fn is_decisive(self) -> bool {
        self == Outcome::Checkmate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn outcome_is_either_draw_or_decisive(o: Outcome) {
        assert_ne!(o.is_draw(), o.is_decisive());
    }
}
