# Changelog
All notable changes to this project will be documented in this file.

The format follows Keep a Changelog and Semantic Versioning.

## [0.0.1] - 2026-02-03
### Added
- Initial experimental release to TestPyPI/PyPI.
- Package name `firepype` with CLI entry points:
  - `firepype` – main pipeline runner (AB-pair NIR reduction).
  - `firepype-telluric` – telluric/response helper.
- Core dependencies: numpy, scipy, astropy, matplotlib.
- Basic project scaffolding: pyproject.toml, README, LICENSE.

### Known limitations
- Tested on a limited sample of FIRE Prism-mode observations.
- Results not guaranteed; validate against FireHose_v2 where possible.
