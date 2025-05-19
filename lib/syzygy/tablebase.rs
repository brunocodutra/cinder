use crate::chess::{Castles, MovePack, PackedMoves, Position};
use crate::syzygy::*;
use crate::util::Integer;
use derive_more::with_trait::Debug;
use std::{collections::HashMap, ffi::OsStr, fs, io, path::Path, str::FromStr};

/// Syzygy tables are available for up to 7 pieces.
pub const MAX_PIECES: usize = 7;

/// A collection of tables.
#[derive(Debug, Default)]
pub struct Tablebase {
    max_pieces: usize,
    wdl: HashMap<NormalizedMaterial, WdlTable>,
    dtz: HashMap<NormalizedMaterial, DtzTable>,
}

impl Tablebase {
    /// Load all relevant tables from a directory.
    pub fn new(path: &Path) -> io::Result<Tablebase> {
        let mut tablebase = Tablebase::default();

        for entry in fs::read_dir(path)? {
            let file = entry?.path();

            let Some(stem) = file.file_stem().and_then(OsStr::to_str) else {
                continue;
            };

            let Ok(material) = Material::from_str(stem) else {
                continue;
            };

            let pieces = material.count();
            if pieces > MAX_PIECES {
                continue;
            }

            let Some(ext) = file.extension() else {
                continue;
            };

            let material = material.normalize();

            if ext == Wdl::EXTENSION {
                let table = WdlTable::new(file, material)?;
                tablebase.wdl.insert(material, table);
            } else if ext == Dtz::EXTENSION {
                let table = DtzTable::new(file, material)?;
                tablebase.dtz.insert(material, table);
            } else {
                continue;
            }

            tablebase.max_pieces = tablebase.max_pieces.max(pieces);
        }

        Ok(tablebase)
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
        if pos.occupied().len() == 2 {
            Some(Wdl::Draw)
        } else {
            let material = Material::from_iter(pos.iter().map(|(p, _)| p)).normalize();
            Some(self.wdl.get(&material)?.probe(pos).unwrap())
        }
    }

    fn get_dtz(&self, pos: &Position, wdl: Wdl) -> Option<Dtz> {
        let material = Material::from_iter(pos.iter().map(|(p, _)| p)).normalize();
        let plies = self.dtz.get(&material)?.probe(pos, wdl).unwrap()?;
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
    moves: PackedMoves,
    wdl: Wdl,
}

impl<'a> ProbeResult<'a> {
    fn zeroing(tb: &'a Tablebase, pos: &'a Position, moves: PackedMoves, wdl: Wdl) -> Self {
        Self {
            kind: ProbeResultKind::Zeroing,
            tb,
            pos,
            moves,
            wdl,
        }
    }

    fn maybe_not_zeroing(
        tb: &'a Tablebase,
        pos: &'a Position,
        moves: PackedMoves,
        wdl: Wdl,
    ) -> Self {
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

        let is_pawn_push = |ms: &MovePack| {
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
        let is_not_zeroing = |ms: &MovePack| ms.is_quiet() && !is_pawn_push(ms);
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
