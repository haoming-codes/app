"""Command line entry point for correcting ASR output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from .lexicon import LexiconEntry
from .matcher import CorrectionEngine


def load_lexicon(path: Path) -> List[LexiconEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: List[LexiconEntry] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                entries.append(LexiconEntry(item))
            elif isinstance(item, dict) and "surface" in item:
                entries.append(LexiconEntry(item["surface"], metadata=item.get("metadata")))
            else:
                raise ValueError(f"Unsupported lexicon entry: {item!r}")
    else:
        raise ValueError("Lexicon JSON must be a list of strings or objects")
    return entries


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lexicon", type=Path, help="Path to the lexicon JSON file")
    parser.add_argument(
        "text",
        nargs="?",
        help="ASR output to correct. If omitted, text is read from standard input.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Similarity threshold in [0,1].",
    )
    parser.add_argument(
        "--window-expansion",
        type=int,
        default=1,
        help="Allow substrings within +/- this many characters of the lexicon length.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_lexicon(args.lexicon)
    engine = CorrectionEngine(
        entries,
        threshold=args.threshold,
        window_expansion=args.window_expansion,
    )
    text = args.text
    if text is None:
        text = sys.stdin.read()
    corrected = engine.correct(text.strip())
    sys.stdout.write(corrected)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
