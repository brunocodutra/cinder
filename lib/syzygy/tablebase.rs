use crate::chess::{Castles, MovePack, MoveSet, Position};
use crate::syzygy::{Dtz, DtzTable, Material, NormalizedMaterial, TableDescriptor, Wdl, WdlTable};
use crate::util::{Integer, Memory};
use derive_more::with_trait::Debug;
use rustc_hash::FxHashMap;
use std::{collections::hash_map::Entry, ffi::OsStr, io, path::Path, str::FromStr};

/// Syzygy tables are available for up to 7 pieces.
pub const MAX_PIECES: usize = 7;

/// A collection of tables.
#[derive(Debug, Default)]
pub struct Tablebase {
    max_pieces: usize,
    wdl: FxHashMap<NormalizedMaterial, WdlTable>,
    dtz: FxHashMap<NormalizedMaterial, DtzTable>,
}

impl Tablebase {
    /// Load Syzygy table from a file.
    pub fn load(&mut self, file: &Path) -> io::Result<()> {
        let Some(stem) = file.file_stem().and_then(OsStr::to_str) else {
            return Err(io::ErrorKind::InvalidFilename.into());
        };

        let Some(ext) = file.extension().and_then(OsStr::to_str) else {
            return Err(io::ErrorKind::InvalidFilename.into());
        };

        let Ok(material) = Material::from_str(stem) else {
            return Err(io::ErrorKind::InvalidFilename.into());
        };

        let pieces = material.count();
        let material = material.normalize();

        if pieces > MAX_PIECES {
            return Err(io::ErrorKind::InvalidData.into());
        }

        if ext.to_lowercase() == Wdl::EXTENSION {
            if let Entry::Vacant(e) = self.wdl.entry(material) {
                e.insert(WdlTable::new(file, material)?);
            }
        } else if ext.to_lowercase() == Dtz::EXTENSION {
            if let Entry::Vacant(e) = self.dtz.entry(material) {
                e.insert(DtzTable::new(file, material)?);
            }
        } else {
            return Err(io::ErrorKind::Unsupported.into());
        }

        self.max_pieces = self.max_pieces.max(pieces);

        Ok(())
    }

    /// Returns the maximum number of pieces across all added tables.
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    /// Probes the tablebase for [`Wdl`] and `[Dtz]` values.
    ///
    /// There are two complications:
    ///
    /// 1. Resolving en passant captures.
    /// 2. When a position has a capture that achieves a particular result
    ///    (e.g., there is a winning capture), then the position itself
    ///    should have at least that value (e.g., it is winning). In this
    ///    case the table can store an arbitrary lower value, whichever is
    ///    best for compression.
    ///
    ///    If the best move is zeroing, then we need remember this to avoid
    ///    probing the DTZ tables.
    pub fn probe<'a>(&'a self, pos: &'a Position) -> Option<ProbeResult<'a>> {
        if pos.castles() != Castles::none() {
            return None;
        }

        // Track best values for en passant and regular captures separately.
        let mut best_capture = Wdl::Loss;
        let mut best_ep = Wdl::Loss;

        let moves = pos.moves();
        let mut all_are_en_passant = None;
        for m in moves.unpack_if(|m| m.is_capture()) {
            let is_pawn = pos.pawns(pos.turn()).contains(m.whence());
            let is_en_passant = pos.en_passant() == Some(m.whither()) && is_pawn;
            all_are_en_passant = Some(all_are_en_passant.unwrap_or(true) && is_en_passant);

            let mut next = pos.clone();
            next.play(m);

            match -self.probe_ab(&next, Wdl::Loss, -best_capture)? {
                v if is_en_passant => best_ep = v.max(best_ep),
                v => best_capture = v.max(best_capture),
            }

            // Early exit if we found a winning capture.
            if best_ep.max(best_capture) == Wdl::Win {
                break;
            }
        }

        if best_ep.max(best_capture) == Wdl::Win {
            return Some(ProbeResult::zeroing(self, pos, moves, Wdl::Win));
        }

        // max(v, best_capture) is the true WDL value of the position,
        // if it has no ep rights.
        let v = self.get_wdl(pos)?;

        // Play the best en passant move if it is strictly better,
        // or if the position would otherwise be stalemate
        if best_ep > v.max(best_capture) || (all_are_en_passant == Some(true) && v == Wdl::Draw) {
            return Some(ProbeResult::zeroing(self, pos, moves, best_ep));
        }

        best_capture = best_ep.max(best_capture);
        if best_capture >= v && best_capture > Wdl::Draw {
            Some(ProbeResult::zeroing(self, pos, moves, best_capture))
        } else {
            let wdl = v.max(best_capture);
            Some(ProbeResult::maybe_not_zeroing(self, pos, moves, wdl))
        }
    }

    fn probe_ab(&self, pos: &Position, mut alpha: Wdl, beta: Wdl) -> Option<Wdl> {
        for m in pos.moves().unpack_if(|m| m.is_capture()) {
            let mut next = pos.clone();
            next.play(m);
            alpha = alpha.max(-self.probe_ab(&next, -beta, -alpha)?);
            if alpha >= beta {
                return Some(alpha);
            }
        }

        Some(self.get_wdl(pos)?.max(alpha))
    }

    fn get_wdl(&self, pos: &Position) -> Option<Wdl> {
        #[ctor::ctor]
        static CACHE: Memory<Wdl, u32> = { Memory::new(1 << 22) };

        if pos.occupied().len() == 2 {
            Some(Wdl::Draw)
        } else if let Some(wdl) = CACHE.get(pos.zobrist()) {
            Some(wdl)
        } else {
            let material = Material::from_iter(pos.iter().map(|(p, _)| p));
            let key = material.normalize();
            let wdl = self.wdl.get(&key)?.probe(pos, material);
            CACHE.set(pos.zobrist(), wdl);
            Some(wdl)
        }
    }

    fn get_dtz(&self, pos: &Position, wdl: Wdl) -> Option<Dtz> {
        let material = Material::from_iter(pos.iter().map(|(p, _)| p));
        let key = material.normalize();
        let plies = self.dtz.get(&key)?.probe(pos, material, wdl)?;
        Some(Dtz::from(wdl).stretch(plies))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum ProbeResultKind {
    /// The best move is zeroing.
    Zeroing,
    /// The best move may not be zeroing.
    MaybeNotZeroing,
}

/// Information about a [`Position`]'s [`Wdl`] and [`Dtz`].
#[derive(Debug)]
pub struct ProbeResult<'a> {
    kind: ProbeResultKind,
    tb: &'a Tablebase,
    pos: &'a Position,
    moves: MovePack,
    wdl: Wdl,
}

impl<'a> ProbeResult<'a> {
    fn zeroing(tb: &'a Tablebase, pos: &'a Position, moves: MovePack, wdl: Wdl) -> Self {
        Self {
            kind: ProbeResultKind::Zeroing,
            tb,
            pos,
            moves,
            wdl,
        }
    }

    fn maybe_not_zeroing(tb: &'a Tablebase, pos: &'a Position, moves: MovePack, wdl: Wdl) -> Self {
        Self {
            kind: ProbeResultKind::MaybeNotZeroing,
            tb,
            pos,
            moves,
            wdl,
        }
    }

    /// The [`Wdl`] if [`Position::halfmoves`] were zero.
    pub fn wdl_after_zeroing(&self) -> Wdl {
        self.wdl
    }

    /// The true [`Wdl`] of the [`Position`].
    pub fn wdl(&self) -> Option<Wdl> {
        match self.pos.halfmoves() {
            n @ 1.. => Some(self.dtz()?.stretch(n as _).into()),
            0 => Some(self.wdl),
        }
    }

    /// The [`Dtz`] of the [`Position`].
    pub fn dtz(&self) -> Option<Dtz> {
        if self.kind == ProbeResultKind::Zeroing || self.wdl == Wdl::Draw {
            return Some(self.wdl.into());
        }

        let is_pawn_push = |ms: &MoveSet| {
            let turn = self.pos.turn();
            !ms.is_capture() && self.pos.pawns(turn).contains(ms.whence())
        };

        // If winning, check for a winning pawn move.
        if self.wdl > Wdl::Draw {
            for m in self.moves.unpack_if(is_pawn_push) {
                let mut next = self.pos.clone();
                next.play(m);
                if -self.tb.probe(&next)?.wdl_after_zeroing() == self.wdl {
                    return Some(self.wdl.into());
                }
            }
        }

        // Probe the DTZ table for a value, if available.
        let mut best = if let Some(dtz) = self.tb.get_dtz(self.pos, self.wdl) {
            return Some(dtz);
        } else if self.wdl < Wdl::CursedWin {
            Some(self.wdl.into())
        } else {
            None
        };

        // Otherwise, do a 1-ply search to find the best DTZ.
        let is_not_zeroing = |ms: &MoveSet| ms.is_quiet() && !is_pawn_push(ms);
        for m in self.moves.unpack_if(is_not_zeroing) {
            let mut next = self.pos.clone();
            next.play(m);
            let v = -self.tb.probe(&next)?.dtz()?;
            if v == Dtz::new(1) && next.is_checkmate() {
                best = Some(Dtz::new(1));
            } else if v.signum() == self.wdl.signum() as i16 {
                let v = v.stretch(1);
                best = match best {
                    Some(best) => Some(v.min(best)),
                    None => Some(v),
                };
            }
        }

        best
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::syzygy::tests::{KNVK_RTBW, KNVKN_RTBZ};
    use std::{fmt::Debug, fs::File, io::Write};
    use tempfile::TempDir;
    use test_strategy::proptest;

    #[proptest(cases = 1)]
    fn loads_wdl_table() {
        let tmp = TempDir::new()?;
        let file = tmp.path().join("KNvK.rtbw");
        File::create(&file)?.write_all(KNVK_RTBW)?;

        let mut tablebase = Tablebase::default();
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));
        assert_eq!(tablebase.wdl.len(), 1);
        assert_eq!(tablebase.dtz.len(), 0);

        let material = Material::from_str("KNvK")?.normalize();
        assert!(tablebase.wdl.contains_key(&material));
    }

    #[proptest(cases = 1)]
    fn loads_dtz_table() {
        let tmp = TempDir::new()?;
        let file = tmp.path().join("KNvKN.rtbz");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;

        let mut tablebase = Tablebase::default();
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));
        assert_eq!(tablebase.wdl.len(), 0);
        assert_eq!(tablebase.dtz.len(), 1);

        let material = Material::from_str("KNvKN")?.normalize();
        assert!(tablebase.dtz.contains_key(&material));
    }

    #[proptest(cases = 1)]
    fn load_ignores_extension_casing() {
        let tmp = TempDir::new()?;

        let mut tablebase = Tablebase::default();

        let file = tmp.path().join("KNvK.RTBW");
        File::create(&file)?.write_all(KNVK_RTBW)?;
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));

        let file = tmp.path().join("KNvKN.RTBZ");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));

        assert_eq!(tablebase.wdl.len(), 1);
        assert_eq!(tablebase.dtz.len(), 1);
    }

    #[proptest(cases = 1)]
    fn load_skips_repeated_table() {
        let tmp = TempDir::new()?;

        let mut tablebase = Tablebase::default();

        let file = tmp.path().join("KNvK.rtbw");
        File::create(&file)?.write_all(KNVK_RTBW)?;
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));

        let file = tmp.path().join("KNvKN.rtbz");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));

        let file = tmp.path().join("KvKN.rtbw");
        File::create(&file)?;
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));

        let file = tmp.path().join("KNvNK.rtbz");
        File::create(&file)?;
        assert_eq!(tablebase.load(&file).map_err(|e| e.kind()), Ok(()));

        assert_eq!(tablebase.wdl.len(), 1);
        assert_eq!(tablebase.dtz.len(), 1);
    }

    #[proptest(cases = 1)]
    fn loading_larger_table_increases_max_pieces() {
        let tmp = TempDir::new()?;
        let mut tablebase = Tablebase::default();

        let file = tmp.path().join("KNvK.rtbw");
        File::create(&file)?.write_all(KNVK_RTBW)?;

        tablebase.load(&file)?;
        assert_eq!(tablebase.max_pieces(), 3);

        let file = tmp.path().join("KNvKN.rtbz");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;

        tablebase.load(&file)?;
        assert_eq!(tablebase.max_pieces, 4);

        assert_eq!(tablebase.wdl.len(), 1);
        assert_eq!(tablebase.dtz.len(), 1);
    }

    #[proptest]
    fn loading_smaller_table_does_not_decrease_max_pieces() {
        let tmp = TempDir::new()?;
        let mut tablebase = Tablebase::default();

        let file = tmp.path().join("KNvKN.rtbz");
        File::create(&file)?.write_all(KNVKN_RTBZ)?;

        tablebase.load(&file)?;
        assert_eq!(tablebase.max_pieces, 4);

        let file = tmp.path().join("KNvK.rtbw");
        File::create(&file)?.write_all(KNVK_RTBW)?;

        tablebase.load(&file)?;
        assert_eq!(tablebase.max_pieces(), 4);

        assert_eq!(tablebase.wdl.len(), 1);
        assert_eq!(tablebase.dtz.len(), 1);
    }

    #[proptest]
    fn loading_file_with_empty_stem_fails(#[strategy("[.][A-Za-z0-9_]{1,10}")] ext: String) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(ext);
        File::create(&file)?;

        let mut tablebase = Tablebase::default();

        assert_eq!(
            tablebase.load(&file).map_err(|e| e.kind()),
            Err(io::ErrorKind::InvalidFilename)
        );
    }

    #[proptest]
    fn loading_file_with_empty_extension_fails(#[strategy("[A-Za-z0-9_]{1,10}[.]?")] stem: String) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(stem);
        File::create(&file)?;

        let mut tablebase = Tablebase::default();

        assert_eq!(
            tablebase.load(&file).map_err(|e| e.kind()),
            Err(io::ErrorKind::InvalidFilename)
        );
    }

    #[proptest]
    fn loading_file_with_invalid_material_fails(
        #[filter(Material::from_str(&#stem).is_err())]
        #[strategy("[A-Za-z0-9_]{1,10}")]
        stem: String,
        #[strategy("(rtbw)|(rtbz)")] ext: String,
    ) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(format!("{stem}.{ext}"));
        File::create(&file)?;

        let mut tablebase = Tablebase::default();

        assert_eq!(
            tablebase.load(&file).map_err(|e| e.kind()),
            Err(io::ErrorKind::InvalidFilename)
        );
    }

    #[proptest]
    fn loading_file_with_too_many_pieces_fails(
        #[strategy("[PNBRQ]{6,}")] stem: String,
        #[strategy("(rtbw)|(rtbz)")] ext: String,
    ) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(format!("K{stem}vK.{ext}"));
        File::create(&file)?;

        let mut tablebase = Tablebase::default();

        assert_eq!(
            tablebase.load(&file).map_err(|e| e.kind()),
            Err(io::ErrorKind::InvalidData)
        );
    }

    #[proptest]
    fn loading_file_with_unsupported_extension_fails(
        #[strategy("[PNBRQ]{1,5}")] stem: String,
        #[filter(!#ext.to_lowercase().starts_with("rtb"))]
        #[strategy("[A-Za-z0-9_]{1,10}")]
        ext: String,
    ) {
        let tmp = TempDir::new()?;
        let file = tmp.path().join(format!("K{stem}vK.{ext}"));
        File::create(&file)?;

        let mut tablebase = Tablebase::default();

        assert_eq!(
            tablebase.load(&file).map_err(|e| e.kind()),
            Err(io::ErrorKind::Unsupported)
        );
    }
}
