#!/usr/bin/env python3
"""Prepare DPR passage document data for the NQ-Open TP-CRAG rerun."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="facebook/wiki_dpr")
    parser.add_argument("--config", default="psgs_w100.nq.compressed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-file", type=Path, default=Path("datasets/nq_open/intermediate/dpr_doc_data.pkl"))
    parser.add_argument("--limit", type=int, help="Optional smoke-test cap.")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split)
    rows = []
    for index, item in enumerate(ds):
        rows.append({
            "id": str(item.get("id", index)),
            "title": item.get("title") or "Unknown",
            "text": item.get("text") or "",
            "source_file": f"{args.dataset}/{args.config}/{args.split}",
        })
        if args.limit is not None and len(rows) >= args.limit:
            break

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("wb") as f:
        pickle.dump(rows, f)
    print(f"Saved {len(rows):,} DPR passages to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
