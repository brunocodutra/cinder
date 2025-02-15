use crate::chess::{Move, Position};

/// A trait for types that record ratings for [`Move`]s.
pub trait Rating {
    /// The [`Move`] bonus rating.
    type Bonus;

    /// Returns the [`Self::Bonus`] rating for a [`Move`].
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus;

    /// Update the [`Self::Bonus`] rating for a [`Move`].
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus);
}

impl<T: Rating> Rating for &T {
    type Bonus = T::Bonus;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        (*self).get(pos, m)
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        (*self).update(pos, m, bonus)
    }
}

impl<T: Rating<Bonus: Default>> Rating for Option<T> {
    type Bonus = T::Bonus;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> Self::Bonus {
        self.as_ref()
            .map_or_else(Default::default, |g| g.get(pos, m))
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, bonus: Self::Bonus) {
        if let Some(g) = self {
            g.update(pos, m, bonus);
        }
    }
}
