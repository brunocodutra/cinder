use crate::chess::{Moves, Position};
use crate::search::{Line, Ply, Pv};
use crate::util::Integer;
use std::path::Path;

mod dtz;
mod fs;
mod material;
mod table;
mod tablebase;
mod wdl;

pub use dtz::*;
pub use fs::*;
pub use material::*;
pub use table::*;
pub use tablebase::*;
pub use wdl::*;

/// The [`Syzygy`] tablebases.
#[derive(Debug)]
pub struct Syzygy {
    tablebase: Tablebase,
}

impl Syzygy {
    /// Initializes the the tablebase from the files in `path`.
    pub fn new(path: &Path) -> Self {
        Self {
            tablebase: Tablebase::new(path).unwrap_or_default(),
        }
    }

    /// The maximum number of pieces available in the tablebase.
    pub fn max_pieces(&self) -> usize {
        self.tablebase.max_pieces()
    }

    /// This [`Position`]'s [`Wdl`].
    pub fn wdl(&self, pos: &Position) -> Option<Wdl> {
        if pos.occupied().len() <= self.max_pieces() {
            Some(self.tablebase.probe(pos)?.wdl_after_zeroing())
        } else {
            None
        }
    }

    /// This [`Position`]'s best [`Pv`].
    pub fn best<T>(&self, pos: &Position, moves: &Moves<T>) -> Option<Pv<1>> {
        if pos.occupied().len() > self.max_pieces() {
            return None;
        }

        let mut best_wdl = Wdl::upper();
        let mut best_gaining = false;
        let mut best_dtz = Dtz::lower();
        let mut best_move = None;

        for &(m, _) in moves {
            let mut next = pos.clone();
            next.play(m);
            let probe = self.tablebase.probe(&next)?;
            let wdl = probe.wdl_after_zeroing();
            if wdl > best_wdl {
                continue;
            } else if wdl == Wdl::Loss && next.is_checkmate() {
                let score = -wdl.to_score(Ply::new(0));
                return Some(Pv::new(score, Line::singular(m)));
            }

            let gaining = !m.is_quiet();
            let dtz = probe.dtz()?.stretch(1);
            if best_move.is_none() || wdl < best_wdl {
                (best_wdl, best_gaining, best_dtz, best_move) = (wdl, gaining, dtz, Some(m));
            } else if (gaining, dtz) > (best_gaining, best_dtz) {
                (best_gaining, best_dtz, best_move) = (gaining, dtz, Some(m));
            }
        }

        let score = -best_wdl.to_score(Ply::new(0));
        Some(Pv::new(score, Line::singular(best_move?)))
    }
}
