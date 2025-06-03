use crate::params::{PARAMS, Params};
use crate::util::Integer;
use derive_more::with_trait::{Display, Error};
use ron::de::{SpannedError, from_str as deserialize};
use ron::ser::to_writer as serialize;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug, Formatter};
use std::str::FromStr;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "spsa", serde(try_from = "i32", into = "i32"))]
#[repr(transparent)]
pub struct Param<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32> {
    #[cfg_attr(test, strategy(MIN..=MAX))]
    value: i32,
}

impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32>
    Param<VALUE, MIN, MAX, BASE>
{
    pub const fn new() -> Self {
        const { assert!(MIN <= VALUE && VALUE <= MAX) }
        Self { value: VALUE }
    }

    pub fn range(&self) -> f64 {
        (MAX - MIN + 1) as f64
    }
}

unsafe impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32> Integer
    for Param<VALUE, MIN, MAX, BASE>
{
    type Repr = i32;
    const MIN: Self::Repr = MIN;
    const MAX: Self::Repr = MAX;
}

impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32> Default
    for Param<VALUE, MIN, MAX, BASE>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32>
    From<Param<VALUE, MIN, MAX, BASE>> for i32
{
    fn from(param: Param<VALUE, MIN, MAX, BASE>) -> Self {
        param.get()
    }
}

/// The reason why constructing [`Param`] from a floating point failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("expected integer in the range `{MIN}..={MAX}`")]
pub struct ParameterOutOfRange<const MIN: i32, const MAX: i32>;

impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32> TryFrom<i32>
    for Param<VALUE, MIN, MAX, BASE>
{
    type Error = ParameterOutOfRange<MIN, MAX>;

    fn try_from(int: i32) -> Result<Self, Self::Error> {
        int.convert().ok_or(ParameterOutOfRange)
    }
}

impl Params {
    pub fn init(self) {
        unsafe { *PARAMS.get().as_mut_unchecked() = self }
    }
}

impl Display for Params {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        serialize(f, self).map_err(|_| fmt::Error)
    }
}

impl FromStr for Params {
    type Err = SpannedError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        deserialize(s)
    }
}

#[cold]
#[ctor::ctor]
#[inline(never)]
unsafe fn init() {
    Params::init(Default::default());
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn parsing_printed_params_is_an_identity(p: Params) {
        assert_eq!(p.to_string().parse(), Ok(p));
    }
}
