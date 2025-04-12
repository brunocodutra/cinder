#![allow(clippy::collapsible_if)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_mm_shuffle))]
#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_aarch64_prefetch, stdarch_neon_dotprod)
)]
#![feature(
    array_chunks,
    coverage_attribute,
    fn_traits,
    gen_blocks,
    impl_trait_in_assoc_type,
    new_zeroed_alloc,
    ptr_as_ref_unchecked,
    round_char_boundary,
    sync_unsafe_cell,
    unboxed_closures
)]

/// Chess domain types.
pub mod chess;
/// Neural network for position evaluation.
pub mod nnue;
/// Minimax searching algorithm.
pub mod search;
/// UCI protocol.
pub mod uci;
/// Assorted utilities.
pub mod util;
