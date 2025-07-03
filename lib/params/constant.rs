use crate::params::Param;
use derive_more::with_trait::Constructor;

macro_rules! define_param {
    ($param:ident, $type:ty) => {
        #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Constructor)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        pub struct $param<const V: $type> {}

        impl<const V: $type> Param for $param<V> {
            type Value = $type;

            #[inline(always)]
            fn get(&self) -> Self::Value {
                V
            }
        }
    };
}

define_param!(Constant, i32);
define_param!(Vector1, [i32; 1]);
define_param!(Vector2, [i32; 2]);
define_param!(Vector3, [i32; 3]);
define_param!(Vector4, [i32; 4]);
define_param!(Vector5, [i32; 5]);
define_param!(Vector6, [i32; 6]);
define_param!(Vector7, [i32; 7]);
define_param!(Vector8, [i32; 8]);
define_param!(Vector9, [i32; 9]);
define_param!(Vector10, [i32; 10]);
define_param!(Vector11, [i32; 11]);
define_param!(Vector12, [i32; 12]);
define_param!(Vector13, [i32; 13]);
define_param!(Vector14, [i32; 14]);
define_param!(Vector15, [i32; 15]);

pub type Scalar<const V: i32> = Constant<V>;
