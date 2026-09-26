"""Streaming TSV readers/writers.

Every file in this challenge is tab separated and large (up to 486 MB), so all
readers here are generators - nothing loads a whole source file into a DataFrame.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

Record = Tuple[str, str, str, str]  # entity_id, business_name, business_address, country


def read_records(path: Path, countries: Optional[Set[str]] = None) -> Iterator[Record]:
    """Yield (entity_id, name, address, country) rows, optionally filtered by country.

    Rows that are short (truncated/corrupt) are padded rather than dropped: every
    test Source-1 entity must survive to the submission, so we never silently lose one.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return
        for row in reader:
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if countries is not None and country not in countries:
                continue
            yield eid, name, addr, country


def read_ground_truth(path: Path) -> Iterator[Tuple[str, List[str]]]:
    """Yield (source1_entity_id, [matched ids]) - empty list for singletons."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            s1 = row[0]
            raw = row[1] if len(row) > 1 else ""
            ids = [x for x in raw.split(",") if x] if raw else []
            yield s1, ids


def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    return dict(read_ground_truth(path))


def list_countries(path: Path) -> Set[str]:
    """Distinct country labels in a source file.

    Country is treated as an open set of string labels - the test set contains
    France, which never appears in training, and any future label must flow through
    untouched.
    """
    seen: Set[str] = set()
    for _, _, _, country in read_records(path):
        seen.add(country)
    return seen


def write_id_lists(path: Path, rows: Iterable[Tuple[str, Iterable[str]]], header: Tuple[str, str]) -> int:
    """Write a two-column TSV of (id, comma-joined id list).

    Deduplicates each list while preserving order - duplicate IDs inside one list
    are an automatic submission rejection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"{header[0]}\t{header[1]}\n")
        for key, ids in rows:
            seen: Set[str] = set()
            ordered: List[str] = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
            fh.write(f"{key}\t{','.join(ordered)}\n")
            n += 1
    return n


__all__ = [
    "Record", "read_records", "read_ground_truth", "load_ground_truth",
    "list_countries", "write_id_lists",
]
