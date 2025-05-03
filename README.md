
<div align="center">
<img src="logo.svg" width="250px" alt="Cinder"/>
<h1>• C I N D E R •</h1>
<br>
</div>

## Overview

Cinder is a strong chess engine written in Rust from scratch.
With a playing strength that is far superior to what humans are capable of,
Cinder is primarily developed to play against other engines. It is regularly tested
at the extremely short time control of 1s+10ms per game. The shorter the time control,
the stronger Cinder tends to play.

## Usage

Cinder implements the UCI protocol and should be compatible with most chess graphical user
interfaces (GUI). Users who are familiar with the UCI protocol may also interact with Cinder
directly on a terminal via its command line interface (CLI).

### Example

```
uci
id name Cinder 0.1.4
id author Bruno Dutra
option name Hash type spin default 16 min 0 max 33554432
option name Threads type spin default 1 min 1 max 4096
uciok
go depth 10
info depth 0 time 0 nodes 0 nps 0 score cp 14 pv g1h3
info depth 1 time 0 nodes 5 nps 125502 score cp 14 pv g1f3
info depth 2 time 0 nodes 11 nps 246913 score cp 14 pv g1f3
info depth 3 time 0 nodes 23 nps 435523 score cp 14 pv g1f3 g8f6
info depth 4 time 0 nodes 107 nps 1181667 score cp 18 pv e2e4 g8f6
info depth 5 time 0 nodes 243 nps 1430253 score cp 16 pv c2c4 g8f6 d2d4
info depth 6 time 0 nodes 552 nps 1791050 score cp 16 pv d2d4 e7e6 g1f3 g8f6 c2c4
info depth 7 time 0 nodes 1081 nps 2071361 score cp 22 pv e2e4 e7e6 g1f3 g8f6
info depth 8 time 0 nodes 1632 nps 2226897 score cp 27 pv e2e4 d7d5 e4d5 g8f6 d2d4 f6d5
info depth 9 time 1 nodes 3389 nps 2439786 score cp 20 pv e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6
info depth 10 time 2 nodes 7493 nps 2584023 score cp 23 pv d2d4 d7d5 c2c4 e7e6 g1f3 f8e7 e2e3 g8f6 b1c3
bestmove d2d4
```

## Contribution

Cinder is an open source project and you're very welcome to contribute to this project by
opening [issues] and/or [pull requests][pulls], see [CONTRIBUTING] for general guidelines.

Building Cinder from source currently requires a recent nightly Rust compiler,
[cargo-make], and [cargo-pgo]. To compile binaries optimized for various CPU architectures,
simply run `cargo make --profile production cinder`. The binary artifacts will be placed
under `target/bin/`.

## License

Cinder is distributed under the terms of the GPL-3.0 license, see [LICENSE] for details.

[issues]:           https://github.com/brunocodutra/cinder/issues
[pulls]:            https://github.com/brunocodutra/cinder/pulls

[cargo-make]:       https://crates.io/crates/cargo-make
[cargo-pgo]:        https://crates.io/crates/cargo-pgo

[LICENSE]:          https://github.com/brunocodutra/cinder/blob/master/LICENSE
[CONTRIBUTING]:     https://github.com/brunocodutra/cinder/blob/master/CONTRIBUTING.md
