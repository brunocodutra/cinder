#![allow(clippy::collapsible_if)]
#![cfg_attr(test, recursion_limit = "1024")]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_mm_shuffle))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_prefetch))]
#![feature(
    coverage_attribute,
    gen_blocks,
    ptr_as_ref_unchecked,
    round_char_boundary,
    slice_swap_unchecked,
    sync_unsafe_cell
)]

/// Chess domain types.
pub mod chess;
/// Neural network for position evaluation.
pub mod nnue;
/// Hyper parameters.
pub mod params;
/// Minimax searching algorithm.
pub mod search;
/// Syzygy tablebase probing.
pub mod syzygy;
/// UCI protocol.
pub mod uci;
/// Assorted utilities.
pub mod util;
