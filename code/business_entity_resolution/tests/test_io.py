"""IO tests. The writer here produces the submitted files, so its invariants are
exactly the ones the official validator rejects on."""
from __future__ import annotations

import pytest

from bijection.io_utils import (
    list_countries, load_ground_truth, read_ground_truth, read_records, write_id_lists,
)

S1 = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _w(p, text):
    p.write_text(text, encoding="utf-8")
    return p


def test_read_records_parses_tabs_not_commas(tmp_path):
    f = _w(tmp_path / "s.tsv", S1 +
           "S1-1\tAcme, Inc\t100 Main St, Springfield, IL\tUS\n")
    rows = list(read_records(f))
    assert rows == [("S1-1", "Acme, Inc", "100 Main St, Springfield, IL", "US")]


def test_read_records_country_filter_is_an_open_set(tmp_path):
    """Country must never be hard-coded to {US, India} - France appears only in test,
    and an unseen label must still flow through."""
    f = _w(tmp_path / "s.tsv", S1 +
           "S1-1\tA\tx\tUS\nS1-2\tB\ty\tFrance\nS1-3\tC\tz\tAtlantis\n")
    assert len(list(read_records(f))) == 3
    assert [r[0] for r in read_records(f, {"France"})] == ["S1-2"]
    assert [r[0] for r in read_records(f, {"Atlantis"})] == ["S1-3"]
    assert list_countries(f) == {"US", "France", "Atlantis"}


def test_short_row_is_padded_not_dropped(tmp_path):
    """Losing a Source-1 entity to a truncated line would fail the whole submission."""
    f = _w(tmp_path / "s.tsv", S1 + "S1-1\tAcme\n")
    rows = list(read_records(f))
    assert rows == [("S1-1", "Acme", "", "")]


def test_empty_address_preserved(tmp_path):
    f = _w(tmp_path / "s.tsv", S1 + "S3-1\tRapid Reit LLC\t\tUS\n")
    assert list(read_records(f))[0][2] == ""


def test_header_only_file_yields_nothing(tmp_path):
    assert list(read_records(_w(tmp_path / "s.tsv", S1))) == []


def test_ground_truth_singletons_are_empty_lists(tmp_path):
    f = _w(tmp_path / "gt.tsv",
           "source1_entity_id\tmatched_entity_ids\n"
           "S1-1\tS2-1,S3-2\n"
           "S1-2\t\n"
           "S1-3\tS2-9\n")
    gt = load_ground_truth(f)
    assert gt["S1-1"] == ["S2-1", "S3-2"]
    assert gt["S1-2"] == []            # singleton, not a [""] artefact
    assert gt["S1-3"] == ["S2-9"]


def test_ground_truth_row_with_no_second_column(tmp_path):
    f = _w(tmp_path / "gt.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\n")
    assert list(read_ground_truth(f)) == [("S1-1", [])]


def test_writer_emits_exact_submission_shape(tmp_path):
    out = tmp_path / "m.tsv"
    n = write_id_lists(out, [("S1-1", ["S2-1", "S3-2"]), ("S1-2", [])],
                       ("source1_entity_id", "matched_entity_ids"))
    assert n == 2
    text = out.read_text(encoding="utf-8")
    assert text == ("source1_entity_id\tmatched_entity_ids\n"
                    "S1-1\tS2-1,S3-2\n"
                    "S1-2\t\n")
    assert '"' not in text              # no CSV quoting


def test_writer_dedupes_within_a_list_preserving_order(tmp_path):
    """A duplicate ID inside one list is an automatic submission rejection."""
    out = tmp_path / "m.tsv"
    write_id_lists(out, [("S1-1", ["S2-1", "S3-2", "S2-1", "S3-2"])],
                   ("source1_entity_id", "matched_entity_ids"))
    assert out.read_text(encoding="utf-8").splitlines()[1] == "S1-1\tS2-1,S3-2"


def test_writer_round_trips_through_the_reader(tmp_path):
    out = tmp_path / "m.tsv"
    write_id_lists(out, [("S1-1", ["S2-1"]), ("S1-2", [])],
                   ("source1_entity_id", "matched_entity_ids"))
    assert load_ground_truth(out) == {"S1-1": ["S2-1"], "S1-2": []}


def test_writer_creates_missing_parent_directory(tmp_path):
    out = tmp_path / "deep" / "nested" / "m.tsv"
    write_id_lists(out, [("S1-1", [])], ("source1_entity_id", "matched_entity_ids"))
    assert out.exists()


def test_unicode_survives_the_round_trip(tmp_path):
    f = _w(tmp_path / "s.tsv", S1 +
           "S2-1\tराम मार्केटिंग\tNEW DELHI\tIndia\n"
           "S2-2\tÉtablissements Dëleves\t27 AV KENNEDY\tFrance\n")
    rows = list(read_records(f))
    assert rows[0][1] == "राम मार्केटिंग"
    assert rows[1][1] == "Établissements Dëleves"
