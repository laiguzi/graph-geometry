"""from_dataset tests against the in-repo research data.

Guarded with ``skip`` so the suite still passes if the package is used without
the in-repo ``data`` tree present.
"""

from pathlib import Path

import pytest

from graph_geometry import load

# tests/ -> graph_geometry/ -> data
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _data_available() -> bool:
    """True only if the research data is present AND readable.

    Checks content, not just existence: on iCloud-synced checkouts the files can
    be evicted to dataless placeholders (correct size, empty reads).
    """
    sentinel = DATA_ROOT / "mygraph" / "mygraph_3cycle4edge.edgelist"
    try:
        return bool(sentinel.read_bytes().strip())
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason=f"research data missing or not materialized at {DATA_ROOT}",
)


def test_mygraph_3cycle4edge():
    G = load.from_dataset("mygraph", DATA_ROOT)
    assert G.is_directed()
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 4


def test_mygraph_variant_selection():
    G = load.from_dataset("mygraph", DATA_ROOT, variant="3cycle")
    assert G.number_of_edges() == 3  # 3-cycle, not the 4-edge variant


def test_email_eu_core():
    """Real data: read from ``data/`` if present, else fetched from SNAP."""
    try:
        G = load.from_dataset("email_eu_core", DATA_ROOT)
    except OSError as exc:
        pytest.skip(f"email-Eu-core unavailable offline: {exc}")
    assert G.is_directed()
    assert G.number_of_nodes() == 1005


def test_email_eu_core_is_fetched_once_when_missing(tmp_path, monkeypatch):
    """No network: urlopen serves gzipped bytes; a second load reads the cache."""
    import gzip
    import io

    from graph_geometry.graph import loaders

    payloads = {
        loaders.EMAIL_EU_CORE_URLS["email-Eu-core.txt"]: b"0 1\n1 2\n2 0\n",
        loaders.EMAIL_EU_CORE_URLS["email-Eu-core-department-labels.txt"]: b"0 1\n",
    }
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(gzip.compress(payloads[url]))

    monkeypatch.setattr(loaders.urllib.request, "urlopen", fake_urlopen)
    G = load.from_dataset("email_eu_core", tmp_path)
    assert G.number_of_edges() == 3
    assert sorted(p.name for p in (tmp_path / "email_EU_core").iterdir()) == [
        "email-Eu-core-department-labels.txt", "email-Eu-core.txt"]
    load.from_dataset("email_eu_core", tmp_path)
    assert len(calls) == 2


def test_email_eu_core_download_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    from graph_geometry.graph import loaders

    def failing_urlopen(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr(loaders.urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(OSError, match="Download and gunzip it by hand"):
        load.from_dataset("email_eu_core", tmp_path)
    assert list((tmp_path / "email_EU_core").iterdir()) == []


def test_cora():
    G = load.from_dataset("cora", DATA_ROOT)
    assert G.number_of_nodes() == 2708


def test_undirected_option():
    G = load.from_dataset("mygraph", DATA_ROOT, directed=False)
    assert not G.is_directed()


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load.from_dataset("not_a_dataset", DATA_ROOT)
