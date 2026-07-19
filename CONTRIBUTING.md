# Contributing to EchoBrief

Thanks for your interest in improving EchoBrief. This project follows a strict,
test-first workflow, so contributions are easiest to review when they match it.

## Workflow

- Open an issue first for anything non-trivial, so we can agree on the approach.
- Write the failing test before the implementation (see the existing suites in `tests/`).
- Run the checks locally before opening a PR:
  ```bash
  uv run ruff check . && uv run ruff format --check .
  uv run pytest --cov=core --cov-fail-under=80
  ```
- Keep `core/` free of UI code, and keep public functions type-hinted.
- Use conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`).

## Contributor terms

By submitting a contribution (a pull request, patch, or any other change) to this
project, you agree that:

1. **You have the right to submit it.** The contribution is your original work, or you
   otherwise have the right to submit it under the terms below, and submitting it does
   not violate any agreement or third-party rights.
2. **Inbound license.** You license your contribution to the project and its users under
   the same license as the project (the MIT License; see [LICENSE](LICENSE)).
3. **Relicensing grant.** You also grant the project maintainer a perpetual, worldwide,
   irrevocable, royalty-free license to use, modify, sublicense, and **relicense** your
   contribution, including under different or commercial license terms, as part of the
   project or a derivative of it. You retain the copyright to your contribution.

This lightweight agreement lets the project keep the option of a commercial or
dual-license offering later without having to track down past contributors. It is a
practical starting point, not a substitute for a formal Contributor License Agreement;
a larger project should adopt a reviewed CLA.
