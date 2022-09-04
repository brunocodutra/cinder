use super::Engine;
use crate::chess::{Move, Position};
use crate::search::{Limits, Search};
use async_trait::async_trait;
use derive_more::From;
use std::convert::Infallible;
use test_strategy::Arbitrary;
use tokio::task::block_in_place;
use tracing::{instrument, Span};

/// A chess engine.
#[derive(Debug, Default, Arbitrary, From)]
pub struct Ai<S: Search> {
    strategy: S,
    limits: Limits,
}

impl<S: Search> Ai<S> {
    /// Constructs [`Ai`] with default [`Limits`].
    pub fn new(strategy: S) -> Self {
        Ai::with_config(strategy, Limits::default())
    }

    /// Constructs [`Ai`] with some [`Limits`].
    pub fn with_config(strategy: S, limits: Limits) -> Self {
        Ai { strategy, limits }
    }
}

#[async_trait]
impl<S: Search + Send> Engine for Ai<S> {
    type Error = Infallible;

    #[instrument(level = "debug", skip(self, pos), ret(Display), err, fields(%pos, depth, score))]
    async fn best(&mut self, pos: &Position) -> Result<Move, Self::Error> {
        let pv = block_in_place(|| self.strategy.search::<1>(pos, self.limits));

        if let Some((d, s)) = Option::zip(pv.depth(), pv.score()) {
            Span::current().record("depth", d).record("score", s);
        }

        Ok(*pv.first().expect("expected at least one legal move"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{MockSearch, Pv, Transposition};
    use std::iter::once;
    use test_strategy::proptest;
    use tokio::runtime;

    #[proptest]
    fn new_applies_default_search_limits() {
        assert_eq!(Ai::new(MockSearch::new()).limits, Limits::default());
    }

    #[proptest]
    fn searches_for_best_move(
        l: Limits,
        pos: Position,
        #[filter(#t.draft() > 0)] t: Transposition,
    ) {
        let rt = runtime::Builder::new_multi_thread().build()?;

        let pv: Pv<256> = once(t).collect();

        let mut strategy = MockSearch::new();
        strategy.expect_search().return_const(pv);

        let mut ai = Ai::with_config(strategy, l);
        assert_eq!(rt.block_on(ai.best(&pos))?, t.best());
    }

    #[proptest]
    #[should_panic]
    fn panics_if_there_are_no_moves(l: Limits, pos: Position) {
        let rt = runtime::Builder::new_multi_thread().build()?;

        let mut strategy = MockSearch::new();
        strategy.expect_search().return_const(Pv::default());

        let mut ai = Ai::with_config(strategy, l);
        rt.block_on(ai.best(&pos))?;
    }
}