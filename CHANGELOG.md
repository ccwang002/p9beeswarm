# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [Unreleased]

### Fixed

- Fixed `vipor.tukeyTexture` (used by `method="tukey"` in `geom_quasirandom`/`geom_beeswarm`) applying its periodic "+2" texture boost only to the first 50 output values instead of to every recycled 50-value cycle, causing the point layout to diverge from R's `vipor::tukeyTexture` for groups larger than 50 observations.



## [0.2.1] - 2026-09-15

### Changed

- Documented installation from PyPI and usage in the README.



## [0.2.0] - 2026-09-15

### Added

- Added plotnine beeswarm, quasirandom, and sina `geom_*` and their corresponding `position_*` functions.
- Added Python compatibility modules for the upstream `beeswarm` and `vipor` packages.
- Added Upstream parity tests and side-by-side examples for `ggbeeswarm`.

### Changed

- Added typed APIs, Python 3.10 support, and `uv`-based package builds and testing.

### Fixed

- Corrected beeswarm, quasirandom, and density behavior to match the upstream R implementations.

[Unreleased]: https://github.com/ccwang002/p9beeswarm/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/ccwang002/p9beeswarm/releases/tag/v0.2.1
[0.2.0]: https://github.com/ccwang002/p9beeswarm/releases/tag/v0.2.0
