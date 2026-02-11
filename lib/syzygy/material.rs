use crate::chess::{Color, Flip, Perspective, Piece, Role};
use crate::util::{Int, Num};
use derive_more::with_trait::{Debug, Deref, Display, Error};
use std::fmt::{self, Formatter, Write};
use std::{iter::repeat_n, str::FromStr};

/// A material key.
#[derive(Debug, Default, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Material({self})")]
pub struct Material(
    #[cfg_attr(test, strategy([
        [0..8, 0..11, 0..11, 0..11, 0..10, 1..2u8],
        [0..8, 0..11, 0..11, 0..11, 0..10, 1..2u8]
    ]))]
    [[u8; 6]; 2],
);

impl Material {
    #[inline(always)]
    pub fn left(self, role: Role) -> usize {
        self.0[0][role.cast::<usize>()].into()
    }

    #[inline(always)]
    pub fn right(self, role: Role) -> usize {
        self.0[1][role.cast::<usize>()].into()
    }

    #[inline(always)]
    pub fn is_symmetric(self) -> bool {
        self.0[0] == self.0[1]
    }

    #[inline(always)]
    pub fn count(self) -> usize {
        self.0.iter().flatten().sum::<u8>().into()
    }

    #[inline(always)]
    pub fn has_pawns(self) -> bool {
        self.left(Role::Pawn) > 0 || self.right(Role::Pawn) > 0
    }

    #[inline(always)]
    pub fn unique_pieces(self) -> usize {
        self.0.iter().flatten().filter(|c| **c == 1).count()
    }

    #[inline(always)]
    pub fn min_like_man(self) -> usize {
        let repeated = self.0.iter().flatten().filter(|c| **c > 1);
        repeated.copied().min().unwrap_or(0).into()
    }

    #[inline(always)]
    pub fn normalize(self) -> NormalizedMaterial {
        NormalizedMaterial(self.perspective(Color::from(self.0[0] < self.0[1])))
    }

    #[inline(always)]
    pub fn iter(self) -> impl Iterator<Item = Piece> {
        Color::iter().zip(self.0).flat_map(|(c, s)| {
            let pieces = move |(r, n)| repeat_n(Piece::new(r, c), n as usize);
            Role::iter().zip(s).flat_map(pieces)
        })
    }
}

impl const Flip for Material {
    #[inline(always)]
    fn flip(self) -> Self {
        Material([self.0[1], self.0[0]])
    }
}

impl FromIterator<Piece> for Material {
    #[inline(always)]
    fn from_iter<T: IntoIterator<Item = Piece>>(iter: T) -> Self {
        let mut material = Material::default();
        for piece in iter {
            material.0[piece.color() as usize][piece.role() as usize] += 1;
        }

        material
    }
}

impl Display for Material {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, side) in self.0.into_iter().enumerate() {
            for (role, count) in Role::iter().zip(side).rev() {
                for _ in 0..count {
                    Display::fmt(&Piece::new(role, Color::White), f)?;
                }
            }

            if i == 0 {
                f.write_char('v')?;
            }
        }

        Ok(())
    }
}

/// The reason why parsing [`Material`] failed.
#[derive(Debug, Display, Error)]
#[derive_const(Default, Clone, Eq, PartialEq)]
#[display("failed to parse material")]
pub struct ParseMaterialError;

impl FromStr for Material {
    type Err = ParseMaterialError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut material = Material::default();
        let (left, right) = s.split_once('v').ok_or(ParseMaterialError)?;
        for (i, s) in [left, right].into_iter().enumerate() {
            for s in s.split_inclusive(|_| true) {
                #[expect(clippy::map_err_ignore)]
                let piece: Piece = s.parse().map_err(|_| ParseMaterialError)?;
                material.0[i][piece.role().cast::<usize>()] += 1;
            }
        }

        if material.left(Role::King) != 1 || material.right(Role::King) != 1 {
            Err(ParseMaterialError)
        } else {
            Ok(material)
        }
    }
}

#[derive(Debug, Display, Copy, Hash, Deref)]
#[derive_const(Clone, Eq, PartialEq)]
#[debug("NormalizedMaterial({self})")]
#[display("{_0}")]
pub struct NormalizedMaterial(Material);

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn symmetric_material_implies_piece_count_is_same_on_both_sides(mat: Material) {
        let left = Role::iter().map(|r| mat.left(r)).sum::<usize>();
        let right = Role::iter().map(|r| mat.right(r)).sum::<usize>();
        assert!(!mat.is_symmetric() || left == right);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn count_returns_total_number_of_pieces(mat: Material) {
        let left = Role::iter().map(|r| mat.left(r)).sum::<usize>();
        let right = Role::iter().map(|r| mat.right(r)).sum::<usize>();
        assert_eq!(mat.count(), left + right);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn normalizing_normalized_material_is_an_identity(mat: Material) {
        assert_eq!(mat.normalize().normalize(), mat.normalize());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn collecting_material_is_an_identity(mat: Material) {
        assert_eq!(mat.iter().collect::<Material>(), mat);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_material_is_an_identity(mat: Material) {
        assert_eq!(mat.to_string().parse(), Ok(mat));
    }
}
