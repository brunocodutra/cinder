use crate::search::{Line, Ply, Pv};
use crate::{chess::Position, util::Integer};
use std::{fs::read_dir, path::PathBuf};

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
#[derive(Debug, Default)]
pub struct Syzygy {
    tablebase: Tablebase,
}

impl Syzygy {
    /// Initializes the the tablebase from the files in `path`.
    pub fn new<'a, I: IntoIterator<Item = &'a PathBuf>>(paths: I) -> Self {
        let mut syzygy = Self::default();

        for path in paths {
            if let Ok(directory) = read_dir(path) {
                for entry in directory.flatten() {
                    if syzygy.tablebase.load(&entry.path()).is_err() {
                        continue;
                    };
                }
            }
        }

        syzygy
    }

    /// The maximum number of pieces available in the tablebase.
    pub fn max_pieces(&self) -> usize {
        self.tablebase.max_pieces()
    }

    /// This [`Position`]'s [`Wdl`] if immediately following a zeroing move.
    pub fn wdl_after_zeroing(&self, pos: &Position) -> Option<Wdl> {
        if pos.halfmoves() == 0 {
            self.wdl(pos)
        } else {
            None
        }
    }

    /// This [`Position`]'s [`Wdl`].
    pub fn wdl(&self, pos: &Position) -> Option<Wdl> {
        if pos.occupied().len() <= self.max_pieces() {
            self.tablebase.probe(pos)?.wdl()
        } else {
            None
        }
    }

    /// This [`Position`]'s [`Dtz`].
    pub fn dtz(&self, pos: &Position) -> Option<Dtz> {
        if pos.occupied().len() <= self.max_pieces() {
            self.tablebase.probe(pos)?.dtz()
        } else {
            None
        }
    }

    /// This [`Position`]'s best [`Pv`].
    pub fn best(&self, pos: &Position) -> Option<Pv<1>> {
        if pos.occupied().len() > self.max_pieces() {
            return None;
        }

        let mut best_wdl = Wdl::upper();
        let mut best_dtz = Dtz::lower();
        let mut best_gaining = false;
        let mut best_move = None;

        for m in pos.moves().unpack() {
            let mut next = pos.clone();
            next.play(m);

            let gaining = !m.is_quiet();
            let probe = self.tablebase.probe(&next)?;
            let (wdl, dtz) = if next.halfmoves() == 0 {
                let wdl = probe.wdl_after_zeroing();
                (wdl, wdl.into())
            } else {
                let dtz = probe.dtz()?.stretch(1);
                (dtz.into(), dtz)
            };

            if wdl > best_wdl {
                continue;
            } else if wdl == Wdl::Loss && next.is_checkmate() {
                return Some(Pv::new(Wdl::Win.to_score(Ply::new(0)), Line::singular(m)));
            } else if best_move.is_none() || wdl < best_wdl {
                (best_wdl, best_gaining, best_dtz, best_move) = (wdl, gaining, dtz, Some(m));
            } else if (gaining, dtz) > (best_gaining, best_dtz) {
                (best_gaining, best_dtz, best_move) = (gaining, dtz, Some(m));
            }
        }

        let score = -best_wdl.to_score(Ply::new(0));
        Some(Pv::new(score, Line::singular(best_move?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{File, create_dir_all};
    use std::io::Write;
    use tempfile::TempDir;
    use test_strategy::proptest;

    pub const KNVK_RTBW: &[u8] = &[
        0x71, 0xE8, 0x23, 0x5D, 0x31, 0x00, 0xEE, 0x66, 0x22, 0x00, 0x80, 0x02, 0x80, 0x02, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xB9, 0xEC, 0xCB, 0xFF, 0x19, 0xC6, 0x77, 0x15, 0x8E, 0x92, 0x8B,
        0x1B, 0x64, 0x12, 0x48, 0xB7,
    ];

    pub const KNVKN_RTBZ: &[u8] = &[
        0xD7, 0x66, 0x0C, 0xA5, 0x40, 0x01, 0x0E, 0x0A, 0x06, 0x02, 0x80, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x2D, 0xA6, 0x2B, 0xDA, 0x60, 0xBF, 0xB9, 0x2C, 0xBD, 0x46, 0x72,
        0xDB, 0x2C, 0xD6, 0xB6, 0xC5,
    ];

    #[proptest(cases = 1)]
    fn new_with_empty_paths() {
        let syzygy = Syzygy::new(&[]);
        assert_eq!(syzygy.max_pieces(), 0);
    }

    #[proptest(cases = 1)]
    fn new_with_single_directory() {
        let tmp = TempDir::new()?;

        let file = tmp.path().join("KNvK.rtbw");
        File::create(&file)?.write_all(KNVK_RTBW)?;

        let file = tmp.path().join("KNvKN.rtbz");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;

        let syzygy = Syzygy::new(&[tmp.path().to_owned()]);
        assert_eq!(syzygy.max_pieces(), 4);
    }

    #[proptest(cases = 1)]
    fn new_with_multiple_directories() {
        let tmp = TempDir::new()?;

        let wdl = tmp.path().join("wdl");
        create_dir_all(&wdl)?;
        let file = wdl.join("KNvK.rtbw");
        File::create(&file)?.write_all(KNVK_RTBW)?;

        let dtz = tmp.path().join("dtz");
        create_dir_all(&dtz)?;
        let file = dtz.join("KNvKN.rtbz");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;

        let syzygy = Syzygy::new(&[wdl, dtz]);
        assert_eq!(syzygy.max_pieces(), 4);
    }

    #[proptest]
    fn new_with_nonexistent_directory(#[strategy("[A-Za-z0-9_]{1,10}")] dir: String) {
        let tmp = TempDir::new()?;
        let syzygy = Syzygy::new(&[tmp.path().join(dir)]);
        assert_eq!(syzygy.max_pieces(), 0);
    }

    #[proptest]
    fn new_with_empty_directory(#[strategy("[A-Za-z0-9_]{1,10}")] dir: String) {
        let tmp = TempDir::new()?;
        let dir = tmp.path().join(dir);
        create_dir_all(&dir)?;

        let syzygy = Syzygy::new(&[dir]);
        assert_eq!(syzygy.max_pieces(), 0);
    }

    #[proptest]
    fn new_with_not_a_directory(#[strategy("[A-Za-z0-9_]{1,10}")] file: String) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(file);
        File::create(&file)?;

        let syzygy = Syzygy::new(&[file]);
        assert_eq!(syzygy.max_pieces(), 0);
    }

    #[proptest]
    fn new_with_directory_containing_invalid_file(#[strategy("[A-Za-z0-9_]{1,10}")] file: String) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(file);
        File::create(file)?;

        let syzygy = Syzygy::new(&[tmp.path().to_owned()]);
        assert_eq!(syzygy.max_pieces(), 0);
    }
}
