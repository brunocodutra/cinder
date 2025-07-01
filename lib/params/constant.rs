#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Param<const V: i32, const K: i32 = 1> {}

impl<const V: i32, const K: i32> Param<V, K> {
    #[inline(always)]
    pub const fn new() -> Self {
        const { assert!(V >= 0) }
        Self {}
    }

    #[inline(always)]
    pub const fn get(&self) -> i32 {
        V
    }
}
