"""Smoke tests: package imports cleanly and the seed KB exists."""
from __future__ import annotations


def test_package_imports():
    import src
    import src.config  # noqa: F401


def test_kb_csv_exists():
    from src.config import KB_CSV

    assert KB_CSV.exists(), f"Knowledge base CSV missing at {KB_CSV}"
