from __future__ import annotations

import importlib.metadata

import scrollstats as m


def test_version():
    assert importlib.metadata.version("scrollstats") == m.__version__
