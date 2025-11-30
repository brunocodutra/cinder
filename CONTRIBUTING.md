## Guidelines

* All contributions to Cinder via pull requests are assumed to be [licensed under the GPL-3.0][LICENSE].
* Every code change must be covered by unit tests, use [cargo-llvm-cov] to generate the code coverage report:
  + `cargo llvm-cov --lib --workspace --features spsa,trainer,bullet/cpu --lcov --output-path lcov.info`
* Besides `cargo test`, make sure [Clippy] and [rustfmt] checks also pass before submitting a pull request:
  + `cargo clippy --workspace --all-features --features bullet/cpu -- -D warnings`
  + `cargo fmt --all -- --check`
* Follow [rustsec.org] advisories when introducing new dependencies, use [cargo-audit] to verify:
  + `cargo audit -D`

[LICENSE]:        https://github.com/brunocodutra/cinder/blob/master/LICENSE
[rustsec.org]:    https://rustsec.org/advisories/

[Clippy]:         https://github.com/rust-lang/rust-clippy#usage
[rustfmt]:        https://github.com/rust-lang/rustfmt#quick-start
[cargo-llvm-cov]: https://github.com/taiki-e/cargo-llvm-cov#usage
[cargo-audit]:    https://github.com/RustSec/cargo-audit#installation
