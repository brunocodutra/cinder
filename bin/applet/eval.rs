use anyhow::Error as Anyhow;
use chess::{Color, Fen, Position};
use clap::Parser;
use eval::Evaluator;
use tracing::{info, instrument};

/// Statically evaluates a position.
#[derive(Debug, Parser)]
#[clap(disable_help_flag = true, disable_version_flag = true)]
pub struct Eval {
    /// The position to evaluate in FEN notation.
    fen: Fen,
}

impl Eval {
    #[instrument(level = "trace", skip(self), err)]
    pub async fn execute(self) -> Result<(), Anyhow> {
        let pos: Position = self.fen.try_into()?;

        info!(
            value = %match pos.turn() {
                Color::White => Evaluator::new().eval(&pos),
                Color::Black => -Evaluator::new().eval(&pos),
            },
        );

        Ok(())
    }
}