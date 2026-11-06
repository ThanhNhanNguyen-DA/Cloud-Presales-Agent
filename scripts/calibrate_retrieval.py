"""Choose a relevance cutoff from reviewed candidates, then check a held-out split.

    python -m scripts.calibrate_retrieval reviewed.json --out /tmp/calibration.json

See docs/retrieval-quality.md for the input schema and the scope of the metric.
This command never edits .env and never calls a model or database.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.retrieval.calibration import calibrate_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-accepted", type=int, default=20)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        report = calibrate_threshold(data["queries"], target_precision=args.target_precision,
                                     min_accepted=args.min_accepted, top_k=data.get("top_k", 4))
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    report.update({"run_at": datetime.now(UTC).isoformat(),
                   "pipeline": data.get("pipeline"),
                   "scope": "Reviewed passage selection, not probability of answer correctness."})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["status"])
    if report["selected_threshold"] is not None:
        print(f"Validated candidate setting: RERANK_MIN_SCORE={report['selected_threshold']:g}")
        print("Use only with the same corpus, model, filters and candidate counts.")
    else:
        print("No threshold recommended: add reviewed data or improve retrieval.")


if __name__ == "__main__":
    main()
