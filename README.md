
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
id name Cinder 0.2.0
id author Bruno Dutra
option name Hash type spin default 16 min 0 max 33554432
option name Threads type spin default 1 min 1 max 4096
option name SyzygyPath type string default <empty>
uciok
go depth 8
info depth 0 time 0 nodes 0 nps 0 score cp 14 pv g1h3
info depth 1 time 0 nodes 5 nps 84373 score cp 14 pv g1f3
info depth 2 time 0 nodes 11 nps 165115 score cp 14 pv g1f3
info depth 3 time 0 nodes 24 nps 312825 score cp 14 pv g1f3 g8f6
info depth 4 time 0 nodes 89 nps 801224 score cp 18 pv e2e4 g8f6
info depth 5 time 0 nodes 194 nps 1033008 score cp 11 pv e2e4 e7e5 b1c3
info depth 6 time 0 nodes 360 nps 1377826 score cp 11 pv e2e4 e7e5 b1c3 g8f6
info depth 7 time 0 nodes 819 nps 1647486 score cp 26 pv e2e4 e7e5 g1f3 g8f6 d2d4
info depth 8 time 1 nodes 2933 nps 2159089 score cp 16 pv e2e4 c7c5 d2d4 c5d4 g1f3 b8c6 f3d4
bestmove e2e4
```

## Acknowledgement

Thanks to [Niklas Fiekas] for [shakmaty-syzygy]. Cinder's implementation of the Syzygy tablebases
probing algorithm is based off a fork of this excellent Rust port.

## Contribution

Cinder is an open source project and you're very welcome to contribute to this project by
opening [issues] and/or [pull requests][pulls], see [CONTRIBUTING] for general guidelines.

Building Cinder from source currently requires a recent nightly Rust compiler,
and [cargo-pgo]. To compile binaries optimized for various CPU architectures,
simply run `make release`. The binary artifacts will be placed under `target/bin/`.

## License

Cinder is distributed under the terms of the GPL-3.0 license, see [LICENSE] for details.

[issues]:           https://github.com/brunocodutra/cinder/issues
[pulls]:            https://github.com/brunocodutra/cinder/pulls

[cargo-pgo]:        https://crates.io/crates/cargo-pgo

[LICENSE]:          https://github.com/brunocodutra/cinder/blob/master/LICENSE
[CONTRIBUTING]:     https://github.com/brunocodutra/cinder/blob/master/CONTRIBUTING.md

[Niklas Fiekas]:    https://github.com/niklasf
[shakmaty-syzygy]:  https://github.com/niklasf/shakmaty-syzygy
