use super::Position;
use derive_more::{DebugCustom, Display, Error, From};
use proptest::{prelude::*, sample::Selector};
use shakmaty as sm;
use std::str::FromStr;
use test_strategy::Arbitrary;

/// A representation of the [algebraic notation].
///
/// [algebraic notation]: https://www.chessprogramming.org/Algebraic_Chess_Notation
#[derive(DebugCustom, Display, Clone, Eq, PartialEq, Hash, Arbitrary)]
#[debug(fmt = "San({})", self)]
#[display(fmt = "{}", _0)]
pub struct San(
    #[strategy(any::<(Position, Selector)>().prop_filter_map("end position", |(pos, selector)| {
            let m = selector.try_select(sm::Position::legal_moves(pos.as_ref()))?;
            Some(sm::san::San::from_move(pos.as_ref(), &m))
        })
    )]
    sm::san::San,
);

impl San {
    pub fn null() -> Self {
        San(sm::san::San::Null)
    }
}

/// The reason why the string is not valid FEN.
#[derive(Debug, Display, Clone, Error, From)]
#[display(fmt = "{}", _0)]
pub struct ParseSanError(#[error(not(source))] sm::san::ParseSanError);

impl FromStr for San {
    type Err = ParseSanError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(San(s.parse()?))
    }
}

#[doc(hidden)]
impl From<sm::san::San> for San {
    fn from(san: sm::san::San) -> Self {
        San(san)
    }
}

#[doc(hidden)]
impl From<San> for sm::san::San {
    fn from(san: San) -> Self {
        san.0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn parsing_printed_san_is_an_identity(san: San) {
        assert_eq!(san.to_string().parse().ok(), Some(san));
    }

    #[proptest]
    fn parsing_invalid_san_fails(
        #[by_ref]
        #[filter(#s.parse::<sm::san::San>().is_err())]
        s: String,
    ) {
        assert!(s.parse::<San>().is_err());
    }

    #[proptest]
    fn san_has_an_equivalent_shakmaty_representation(san: San) {
        assert_eq!(San::from(sm::san::San::from(san.clone())), san);
    }
}