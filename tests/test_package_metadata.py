"""Release identity: the version must agree everywhere it is declared.

Regression for an audit finding that `pyproject.toml`, `CITATION.cff` and
`graph_geometry.__version__` disagreed. A DOI and a citation are
attached to a version string, so a package that misreports its own version
cannot be cited reliably.
"""

import re
from pathlib import Path

import graph_geometry as gg

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version in pyproject.toml"
    return match.group(1)


def _citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*([^\s#]+)", text, re.MULTILINE)
    assert match, "no version in CITATION.cff"
    return match.group(1).strip('"\'')


def test_dunder_version_matches_pyproject():
    assert gg.__version__ == _pyproject_version()


def test_citation_version_matches_pyproject():
    assert _citation_version() == _pyproject_version()
