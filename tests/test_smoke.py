"""Scaffolding smoke test — replaced by real suites from Task 2 onward."""

import core


def test_core_package_is_importable() -> None:
    assert core.__doc__ is not None
