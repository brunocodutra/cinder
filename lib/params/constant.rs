#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Param<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32> {}

impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32>
    Param<VALUE, MIN, MAX, BASE>
{
    #[inline(always)]
    pub fn get(&self) -> i32 {
        const { assert!(MIN <= VALUE && VALUE <= MAX) }
        VALUE
    }
}
