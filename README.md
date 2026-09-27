
<div align="center">
    <img src="logo.svg" width="250px" alt="Cinder"/>
    <div>• C I N D E R •</div>
    <br>
    <a href="https://www.sp-cc.de/"><img src="https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fwww.sp-cc.de%2F&amp;search=%26nbsp%3B%5Cs*(%5Cd%2B)%5Cs%2BCinder%5B%5E%3A%5D*%3A%5Cs*(%5Cd%2B)&amp;replace=%23%241%20%C2%B7%20%242&amp;label=SPCC&amp;labelColor=1d1a16&amp;color=cc1f0d&amp;cacheSeconds=86400" alt="Cinder on SPCC UHO-Top15"></a>
    <a href="https://computerchess.org.uk/404/cgi/compare_engines.cgi?class=Single-CPU+engines&amp;print=Rating+list&amp;cross_tables_for_best_versions_only=1"><img src="https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fcomputerchess.org.uk%2F404%2Fcgi%2Fcompare_engines.cgi%3Fclass%3DSingle-CPU%2Bengines%26print%3DRating%2Blist%26cross_tables_for_best_versions_only%3D1&amp;search=class%3D%22number%22%3E%3Cb%3E(%5Cd%2B)%3C%2Fb%3E%3C%2Ftd%3E%5B%5Cs%5CS%5D%7B0%2C400%7D%3F%3ECinder%20%5B%5Cd.%5D%2B%2064-bit%3C%2Fa%3E%3C%2Fb%3E%3C%2Fspan%3E%3C%2Ftd%3E%3Ctd%20rowspan%3D2%20class%3D%22rating%22%3E%3Cb%3E(%5Cd%2B)%3C%2Fb%3E%3C%2Ftd%3E&amp;replace=%23%241%20%C2%B7%20%242&amp;label=CCRL%20Blitz&amp;labelColor=1d1a16&amp;color=cc1f0d&amp;cacheSeconds=86400" alt="Cinder on CCRL Blitz"></a>
    <a href="https://computerchess.org.uk/4040/cgi/compare_engines.cgi?class=Single-CPU+engines&amp;print=Rating+list&amp;cross_tables_for_best_versions_only=1"><img src="https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fcomputerchess.org.uk%2F4040%2Fcgi%2Fcompare_engines.cgi%3Fclass%3DSingle-CPU%2Bengines%26print%3DRating%2Blist%26cross_tables_for_best_versions_only%3D1&amp;search=class%3D%22number%22%3E%3Cb%3E(%5Cd%2B)%3C%2Fb%3E%3C%2Ftd%3E%5B%5Cs%5CS%5D%7B0%2C400%7D%3F%3ECinder%20%5B%5Cd.%5D%2B%2064-bit%3C%2Fa%3E%3C%2Fb%3E%3C%2Fspan%3E%3C%2Ftd%3E%3Ctd%20rowspan%3D2%20class%3D%22rating%22%3E%3Cb%3E(%5Cd%2B)%3C%2Fb%3E%3C%2Ftd%3E&amp;replace=%23%241%20%C2%B7%20%242&amp;label=CCRL%2040%2F15&amp;labelColor=1d1a16&amp;color=cc1f0d&amp;cacheSeconds=86400" alt="Cinder on CCRL 40/15"></a>
    <a href="http://www.cegt.net/40_40%20Rating%20List/40_40%20All%20Versions/rangliste.html"><img src="https://img.shields.io/badge/dynamic/regex?url=http%3A%2F%2Fwww.cegt.net%2F40_40%2520Rating%2520List%2F40_40%2520All%2520Versions%2Frangliste.html&amp;search=%3Ctd%3E(%5Cd%2B)%3C%2Ftd%3E%5Cs*%3Ctd%20class%3D%22left%22%3E%3Ca%20class%3D%22t%22%20href%3D%22%5B%5E%22%5D*%22%3ECinder%5B%5E%3C%5D*%3C%2Fa%3E%3C%2Ftd%3E%5Cs*%3Ctd%3E(%5Cd%2B)%3C%2Ftd%3E&amp;replace=%23%241%20%C2%B7%20%242&amp;label=CEGT%2040%2F20&amp;labelColor=1d1a16&amp;color=cc1f0d&amp;cacheSeconds=86400" alt="Cinder on CEGT 40/20"></a>
</div>

## Overview

Cinder is a hobby chess engine written in Rust.

### Playing Strength

| Version  | [SPCC] | [Ipmanchess] | [CCRL Blitz] | [CCRL 40/15] | [CEGT 40/20] |
|----------|:------:|:------------:|:------------:|:------------:|:------------:|
| [v0.6.*] | 3773   | 3576         | 3763         | 3607         | 3615         |
| [v0.5.*] | -      | 3556         | 3737         | 3602         | 3589         |
| [v0.4.*] | -      | -            | 3682         | 3559         | 3535         |
| [v0.3.*] | -      | -            | 3655         | 3545         | 3512         |
| [v0.2.*] | -      | -            | 3632         | 3522         | 3481         |
| [v0.1.*] | -      | -            | -            | 3496         | 3439         |

## Getting started

### Prebuilt binaries

Prebuilt binaries for various popular platforms and CPU architectures are available on
the [releases] page. Below is a reference for which binary to pick.
The table is ordered from most compatible to most performant.
You should prefer the most performant binary that runs on your machine.

| Suffix      | Description                                                     |
|-------------|-----------------------------------------------------------------|
| `*-sse4`    | Compatible with most Intel and AMD CPUs                         |
| `*-avx2`    | Compatible with Intel Haswell (2013+) and AMD Excavator (2015+) |
| `*-avx512`  | Compatible with Intel Ice Lake (2019+) and AMD Zen 4 (2022+)    |
| `*-neon`    | Compatible with ARMv8.2-A (2017+), including Apple M1 and newer |

### Building from source

Building Cinder from source currently requires a recent nightly Rust compiler.
Run `make` to build a binary optimized for your CPU architecture.
Run `make help` to view all available build targets.
You'll find the binaries under `target/bin/`.

### Usage

Cinder implements the UCI protocol and should be compatible with most chess graphical user
interfaces (GUI). Users who are familiar with the UCI protocol may also interact with Cinder
directly on a terminal via its command line interface (CLI).

#### UCI options

| Name            | Default   | Unit  | Description                                              |
|-----------------|:---------:|-------|----------------------------------------------------------|
| `Hash`          | 16        | MiB   | Memory allocated for the transposition table             |
| `Threads`       | 1         | count | Number of search threads used to search                  |
| `MoveOverhead`  | 10        | ms    | Clock time assumed to be lost to system latency per move |
| `SyzygyPath`    | `<empty>` | -     | Path to a directory containing Syzygy tablebases         |

## Acknowledgement

The efficiently updatable neural networks (NNUE) Cinder uses for position evaluation are
trained with [bullet], by [Jamie Whiting], using data generated by the [Leela Chess Zero]
project, which is available under the [Open Database License] (ODbL)

Cinder's implementation of the Syzygy tablebases probing algorithm is based on a fork of
[shakmaty-syzygy], by [Niklas Fiekas].

## Contribution

Cinder is an open source project and you're very welcome to contribute to this project by
opening [issues] and/or [pull requests][pulls], see [CONTRIBUTING] for general guidelines.

## License

Cinder is distributed under the terms of the GPL-3.0 license, see [LICENSE] for details.

[issues]:                   https://github.com/brunocodutra/cinder/issues
[pulls]:                    https://github.com/brunocodutra/cinder/pulls
[releases]:                 https://github.com/brunocodutra/cinder/releases/latest

[LICENSE]:                  https://github.com/brunocodutra/cinder/blob/master/LICENSE
[CONTRIBUTING]:             https://github.com/brunocodutra/cinder/blob/master/CONTRIBUTING.md

[Niklas Fiekas]:            https://github.com/niklasf
[shakmaty-syzygy]:          https://github.com/niklasf/shakmaty-syzygy
[Jamie Whiting]:            https://github.com/jw1912
[bullet]:                   https://github.com/jw1912/bullet
[Leela Chess Zero]:         https://lczero.org/
[Open Database License]:    https://opendatacommons.org/licenses/odbl/1-0/

[v0.6.*]:                   https://github.com/brunocodutra/cinder/releases/tag/v0.6.1
[v0.5.*]:                   https://github.com/brunocodutra/cinder/releases/tag/v0.5.2
[v0.4.*]:                   https://github.com/brunocodutra/cinder/releases/tag/v0.4.1
[v0.3.*]:                   https://github.com/brunocodutra/cinder/releases/tag/v0.3.1
[v0.2.*]:                   https://github.com/brunocodutra/cinder/releases/tag/v0.2.0
[v0.1.*]:                   https://github.com/brunocodutra/cinder/releases/tag/v0.1.4

[SPCC]:                     https://www.sp-cc.de/
[Ipmanchess]:               https://ipmanchess.yolasite.com/r9-7945hx.php
[CCRL Blitz]:               https://computerchess.org.uk/404/cgi/compare_engines.cgi?class=Single-CPU+engines&print=Rating+list&cross_tables_for_best_versions_only=1
[CCRL 40/15]:               https://computerchess.org.uk/4040/cgi/compare_engines.cgi?class=Single-CPU+engines&print=Rating+list&cross_tables_for_best_versions_only=1
[CEGT 40/20]:               http://www.cegt.net/40_40%20Rating%20List/40_40%20All%20Versions/rangliste.html
