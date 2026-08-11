# Contributing to arwe

Thanks for your interest in contributing! This document covers the basics of
building, testing, and opening pull requests for arwe.

## Development setup

- Rust stable toolchain (rustup recommended). No MSRV is pinned; CI runs on
  stable.
- Build with `cargo build`.
- Generate the demo image with `cargo run --example gen_demo` (writes
  `demo.png`), then try e.g. `cargo run -- matrix --input demo.png`.

## Checks before opening a PR

All of the following run in CI, so please make sure they pass locally first:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

The test suite is fully offline: vision-model behavior is tested against a
local mock server and OCR against a mock `tesseract` binary, so no API keys or
Tesseract installation are needed.

## Adding a subcommand

- CLI argument definitions live in `src/cli.rs`; dispatch lives in
  `src/main.rs`.
- The processing pipeline is load → crop → width → stride → (kernel) → render.
  See the README for the exact order and coordinate semantics (boxes and
  samples are reported in original-image pixel coordinates).
- Add a test in `tests/` for every new user-facing behavior; unit-test pure
  helpers next to the code they cover.
- Update the README subcommand table and section when you add or change a
  command.

## Adding a preset kernel

Add the kernel to `src/kernels.rs` and list it with the existing presets. Keep
the matrix compact (small integer coefficients preferred) and note any
normalization or stride behavior in the README kernel section if it differs
from the defaults.

## Mutation testing

The project uses `cargo-mutants` to find weakly-tested code:

```bash
cargo mutants --copy-target true --jobs 8 --exclude 'examples/*'
```

The goal is to keep surviving mutants limited to semantically equivalent
changes.

## Issues and pull requests

- Use the issue templates for bug reports and feature requests.
- Reference related issues in your PR (`Closes #123`).
- Keep PRs focused: one logical change per PR, with a clear description.
