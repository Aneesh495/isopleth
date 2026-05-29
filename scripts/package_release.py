#!/usr/bin/env python3
"""Release packaging and distribution manifest builder.

Packages verified release artifacts into distribution tarball with SHA-256 manifest:
1. Validates certified evidence bundle exists and passes all verification gates.
2. Audits all source and documentation files for release readiness.
3. Computes cryptographic checksums for all distribution files.
4. Creates distribution archive in dist/ directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.verify_evidence import EvidenceBundleVerifier


def hash_file(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package Isopleth release distribution."
    )
    parser.add_argument(
        "--bundle",
        "-b",
        type=str,
        default="artifacts/evidence_bundle.json",
        help="Path to certified evidence bundle.",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="0.1.0",
        help="Release version string.",
    )
    parser.add_argument(
        "--dist-dir",
        "-d",
        type=str,
        default="dist",
        help="Destination directory for distribution archives.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    bundle_path = project_root / args.bundle
    dist_dir = project_root / args.dist_dir
    dist_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(f"Packaging Isopleth Release v{args.version}")
    print("=================================================================")

    # 1. Audit evidence bundle
    print("\n1. Auditing evidence bundle...")
    verifier = EvidenceBundleVerifier()
    rep = verifier.verify_file(bundle_path)
    if not rep.valid:
        print(f"ERROR: Evidence bundle verification failed with {len(rep.errors)} errors:")
        for err in rep.errors:
            print(f"  - {err}")
        return 1
    print(f"  Certified bundle SHA-256: {rep.bundle_hash}")

    # 2. Collect distribution files
    print("\n2. Cataloging distribution files...")
    include_paths = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src",
        "scripts",
        "docs",
        "artifacts/evidence_bundle.json",
    ]

    manifest: dict[str, str] = {}
    for item in include_paths:
        p = project_root / item
        if p.is_file():
            rel_name = p.relative_to(project_root).as_posix()
            manifest[rel_name] = hash_file(p)
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and "__pycache__" not in sub.parts and not sub.name.endswith(".pyc"):
                    rel_name = sub.relative_to(project_root).as_posix()
                    manifest[rel_name] = hash_file(sub)

    print(f"  Cataloged {len(manifest)} release files.")

    # 3. Write release manifest
    manifest_path = dist_dir / f"isopleth-v{args.version}-manifest.json"
    release_info = {
        "project": "isopleth",
        "version": args.version,
        "evidence_bundle_sha256": rep.bundle_hash,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files_count": len(manifest),
        "files_checksums": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(release_info, f, indent=2)
    print(f"  Wrote release manifest to: {manifest_path.resolve()}")

    # 4. Create release tarball
    archive_name = f"isopleth-v{args.version}.tar.gz"
    archive_path = dist_dir / archive_name
    print(f"\n3. Creating release tarball: {archive_name}...")

    with tarfile.open(archive_path, "w:gz") as tar:
        for rel_file in manifest.keys():
            abs_file = project_root / rel_file
            tar.add(abs_file, arcname=f"isopleth-v{args.version}/{rel_file}")
        tar.add(manifest_path, arcname=f"isopleth-v{args.version}/release_manifest.json")

    archive_hash = hash_file(archive_path)
    print(f"  Archive created: {archive_path.resolve()}")
    print(f"  Archive size:    {archive_path.stat().st_size / 1024:.1f} KB")
    print(f"  Archive SHA-256: {archive_hash}")
    print("=================================================================")
    print("Release Packaging Complete")
    print("=================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
