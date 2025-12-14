#![allow(clippy::collapsible_if, clippy::double_parens, long_running_const_eval)]
#![cfg_attr(test, recursion_limit = "1024")]
#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_aarch64_prefetch, stdarch_neon_dotprod)
)]
#![feature(
    adt_const_params,
    const_cmp,
    const_default,
    const_destruct,
    const_index,
    const_ops,
    const_option_ops,
    const_range,
    const_result_unwrap_unchecked,
    const_trait_impl,
    coverage_attribute,
    gen_blocks,
    iter_array_chunks,
    pointer_is_aligned_to,
    portable_simd,
    ptr_as_ref_unchecked,
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
/// Single Instruction, Multiple Data (SIMD) utilities.
pub mod simd;
/// Syzygy tablebase probing.
pub mod syzygy;
/// UCI protocol.
pub mod uci;
/// Assorted utilities.
pub mod util;
