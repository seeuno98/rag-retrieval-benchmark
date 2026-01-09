"""Inspect MSMARCO BEIR qrels TSV header and first row."""

from __future__ import annotations

import csv
from pathlib import Path


def inspect_qrels(path: Path) -> tuple[list[str], list[str]]:
    """Return header and first row from a qrels TSV file."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        first_row = next(reader)
    return header, first_row


def main() -> None:
    """Print MSMARCO qrels header and first row for quick inspection."""
    qrels_path = Path("data/beir_raw/msmarco/msmarco/qrels/test.tsv")
    header, first_row = inspect_qrels(qrels_path)
    print("header:", header)
    print("first_row:", first_row)


if __name__ == "__main__":
    main()
