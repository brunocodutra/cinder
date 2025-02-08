use crate::chess::{Color, Move, ParsePositionError, Perspective, Piece, Position, Role, Square};
use crate::nnue::{Accumulator, Feature, Material, Nnue, Positional, Value};
use crate::util::{Assume, Integer};
use arrayvec::ArrayVec;
use derive_more::with_trait::{Debug, Deref, Display};
use std::str::FromStr;

#[cfg(test)]
use proptest::prelude::*;

/// An incrementally evaluated [`Position`].
#[derive(Debug, Display, Clone, Eq, PartialEq, Hash, Deref)]
#[debug("Evaluator({self})")]
#[display("{pos}")]
pub struct Evaluator {
    #[deref]
    pos: Position,
    acc: (Material, Positional),
}

#[cfg(test)]
impl Arbitrary for Evaluator {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        any::<Position>().prop_map(Evaluator::new).boxed()
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new(Position::default())
    }
}

impl Evaluator {
    /// Constructs the evaluator from a [`Position`].
    pub fn new(pos: Position) -> Self {
        let mut acc: (Material, Positional) = Default::default();

        for side in Color::iter() {
            let ksq = pos.king(side);
            for (p, s) in pos.iter() {
                acc.add(side, Feature::new(side, ksq, p, s));
            }
        }

        Evaluator { pos, acc }
    }

    /// Play a [null-move].
    ///
    /// [null-move]: https://www.chessprogramming.org/Null_Move
    pub fn pass(&mut self) {
        self.pos.pass();
    }

    /// Play a [`Move`].
    pub fn play(&mut self, m: Move) {
        let turn = self.turn();
        let promotion = m.promotion();
        let (wc, wt) = (m.whence(), m.whither());
        let (role, capture) = self.pos.play(m);
        let mut sides = ArrayVec::<Color, 2>::from([!turn, turn]);

        if role == Role::King
            && Feature::new(turn, wc, Piece::lower(), Square::lower())
                != Feature::new(turn, wt, Piece::lower(), Square::lower())
        {
            sides.truncate(1);
            self.acc.refresh(turn);
            for (p, s) in self.pos.iter() {
                self.acc.add(turn, Feature::new(turn, wt, p, s));
            }
        }

        for side in sides {
            let ksq = self.king(side);
            let old = Feature::new(side, ksq, Piece::new(role, turn), wc);
            let new = Feature::new(side, ksq, Piece::new(promotion.unwrap_or(role), turn), wt);
            self.acc.replace(side, old, new);

            if let Some((r, sq)) = capture {
                let victim = Piece::new(r, !turn);
                self.acc.remove(side, Feature::new(side, ksq, victim, sq));
            } else if role == Role::King && (wt - wc).abs() == 2 {
                let rook = Piece::new(Role::Rook, turn);
                let (wc, wt) = if wt > wc {
                    (Square::H1.perspective(turn), Square::F1.perspective(turn))
                } else {
                    (Square::A1.perspective(turn), Square::D1.perspective(turn))
                };

                let old = Feature::new(side, ksq, rook, wc);
                let new = Feature::new(side, ksq, rook, wt);
                self.acc.replace(side, old, new);
            }
        }
    }

    /// Estimates the material gain of a move.
    pub fn gain(&self, m: Move) -> Value {
        let psqt = Nnue::psqt();
        let turn = self.turn();
        let promotion = m.promotion();
        let (wc, wt) = (m.whence(), m.whither());
        let role = self[wc].assume().role();
        let phase = (self.occupied().len() - 1 - m.is_capture() as usize) / 4;
        let mut deltas = [0i32, 0i32];

        for (delta, side) in deltas.iter_mut().zip([turn, !turn]) {
            let ksq = self.king(side);

            let old = Feature::new(side, ksq, Piece::new(role, turn), wc);
            *delta -= psqt.get(old.cast::<usize>()).assume().get(phase).assume();

            let new = Feature::new(side, ksq, Piece::new(promotion.unwrap_or(role), turn), wt);
            *delta += psqt.get(new.cast::<usize>()).assume().get(phase).assume();

            if m.is_capture() {
                let (victim, target) = match self[wt] {
                    Some(p) => (p, wt),
                    None => (
                        Piece::new(Role::Pawn, !turn),
                        Square::new(wt.file(), wc.rank()),
                    ),
                };

                let cap = Feature::new(side, ksq, victim, target);
                *delta -= psqt.get(cap.cast::<usize>()).assume().get(phase).assume();
            } else if role == Role::King && (wt - wc).abs() == 2 {
                let rook = Piece::new(Role::Rook, turn);
                let (wc, wt) = if wt > wc {
                    (Square::H1.perspective(turn), Square::F1.perspective(turn))
                } else {
                    (Square::A1.perspective(turn), Square::D1.perspective(turn))
                };

                let old = Feature::new(side, ksq, rook, wc);
                *delta -= psqt.get(old.cast::<usize>()).assume().get(phase).assume();

                let new = Feature::new(side, ksq, rook, wt);
                *delta += psqt.get(new.cast::<usize>()).assume().get(phase).assume();
            }
        }

        let value = (deltas[0] - deltas[1]) >> 7;
        value.saturate()
    }

    /// The [`Position`]'s evaluation.
    pub fn evaluate(&self) -> Value {
        let phase = (self.occupied().len() - 1) / 4;
        let value = self.acc.evaluate(self.turn(), phase) >> 7;
        value.saturate()
    }
}

impl FromStr for Evaluator {
    type Err = ParsePositionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s.parse()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::sample::Selector;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn play_updates_evaluator(
        #[filter(#e.outcome().is_none())] mut e: Evaluator,
        #[map(|sq: Selector| sq.select(#e.moves().flatten()))] m: Move,
    ) {
        let mut pos = e.pos.clone();
        e.play(m);
        pos.play(m);
        assert_eq!(e, Evaluator::new(pos));
    }

    #[proptest]
    fn pass_updates_evaluator(#[filter(!#e.is_check())] mut e: Evaluator) {
        let mut pos = e.pos.clone();
        e.pass();
        pos.pass();
        assert_eq!(e, Evaluator::new(pos));
    }

    #[proptest]
    fn parsing_printed_evaluator_is_an_identity(e: Evaluator) {
        assert_eq!(e.to_string().parse(), Ok(e));
    }
}
