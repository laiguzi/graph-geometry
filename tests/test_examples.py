"""Smoke tests that the example scripts keep running."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["quickstart", "balancing_factor"])
def test_example_runs(name, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # any figure output lands in tmp, not the repo
    _load(name).main()
    out = capsys.readouterr().out
    assert out  # produced some output without raising
