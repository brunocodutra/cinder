
<div align="center">
<img src="logo.svg" width="250px" alt="Cinder"/>
<h1>• C I N D E R •</h1>
<br>
</div>

## Overview

Cinder is an independent chess engine written in Rust from scratch.
With a playing strength that is far superior to what humans are capable of,
Cinder is primarily developed to play against other engines. It is regularly tested
at the extremely short time control of 3s+25ms per game. The shorter the time control,
the stronger Cinder tends to play.

## Usage

Cinder implements the UCI protocol and should be compatible with most current
chess graphical user interfaces (GUI). Users who are familiar with the UCI protocol
may also interact with Cinder directly on a terminal via its command line interface (CLI).
In addition to the standard UCI commands, Cinder also implements a custom command `eval`
that prints Cinder's evaluation of the current position in its own internal units.

### Example

```
uci
id name Cinder
id author Bruno Dutra
option name Hash type spin default 16 min 0 max 33554432
option name Threads type spin default 1 min 1 max 65536
uciok
go depth 15
info score cp +17 pv d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 b1c3 f8b4 c4d5 e6d5 c1g5 b4c3 b2c3 h7h6 g5f6
bestmove d2d4
```

## Contribution

Cinder is an open source project and you're very welcome to contribute to this project by
opening [issues] and/or [pull requests][pulls], see [CONTRIBUTING] for general guidelines.

Building Cinder from source currently requires a recent nightly Rust compiler,
[cargo-make], and [cargo-pgo]. To compile binaries optimized for your CPU architecture,
simply run `cargo make cinder`.

## License

Cinder is distributed under the terms of the GPL-3.0 license, see [LICENSE] for details.

[issues]:           https://github.com/brunocodutra/cinder/issues
[pulls]:            https://github.com/brunocodutra/cinder/pulls

[cargo-make]:       https://crates.io/crates/cargo-make
[cargo-pgo]:        https://crates.io/crates/cargo-pgo

[LICENSE]:          https://github.com/brunocodutra/cinder/blob/master/LICENSE
[CONTRIBUTING]:     https://github.com/brunocodutra/cinder/blob/master/CONTRIBUTING.md
