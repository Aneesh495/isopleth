#!/usr/bin/env python3
"""Acceptance campaign execution driver.

Runs all 12 acceptance gates (IA01 through IA12) and exports a certified,
SHA-256 hashed evidence bundle JSON file for independent audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from isopleth.evaluation.acceptance import AcceptanceCampaignRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Isopleth acceptance campaign and export evidence bundle."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="artifacts/evidence_bundle.json",
        help="Target output path for the evidence bundle JSON file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose per-gate progress output.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=================================================================")
        print("Starting Isopleth 12-Gate Acceptance Campaign")
        print("=================================================================")

    t0 = time.perf_counter()
    runner = AcceptanceCampaignRunner()
    report = runner.execute_full_campaign()
    elapsed = time.perf_counter() - t0

    if not args.quiet:
        for outcome in report.outcomes:
            status = "PASS" if outcome.passed else "FAIL"
            print(
                f"[{status}] {outcome.gate_id}: {outcome.gate_name} "
                f"({outcome.wall_clock_seconds:.3f}s)"
            )
            print(f"       {outcome.summary}")
        print("-----------------------------------------------------------------")
        print(
            f"Campaign Result: {report.gates_passed}/{report.total_gates} gates passed "
            f"in {elapsed:.2f}s"
        )
        print(f"SHA-256 Bundle Digest: {report.sha256_bundle_digest}")
        print("=================================================================")

    # Serialize report to JSON
    report_dict = asdict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    if not args.quiet:
        print(f"Wrote certified evidence bundle to: {output_path.resolve()}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
