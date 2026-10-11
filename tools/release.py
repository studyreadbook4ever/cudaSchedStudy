#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Audit independently authored text source and export a reproducible source archive."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {"LICENSE", "README.md", "THIRD_PARTY.md", "CONTRIBUTING.md", "CMakeLists.txt",
       ".gitignore", ".clang-format"}
DIRECTORIES = {"src", "include", "python", "tests", "tools", "scripts", "containers", "docs"}
EXCLUDED = {".git", ".tools", "runtime", "artifacts", "dist", "__pycache__",
            "SOURCE_MANIFEST.json"}  # Generated anew on each export, including re-export.
SUFFIXES = {".cpp", ".cu", ".hpp", ".h", ".py", ".sh", ".md"}
SPECIAL = {"containers/Dockerfile", "docs/images/architecture.svg"}
SECRET_PATTERNS = [re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
                   re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
                   re.compile(rb"github_pat_[A-Za-z0-9_]{30,}")]


def source_files():
    found = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in rel.parts) or (
                rel.parts[0].startswith("build") and (ROOT / rel.parts[0]).is_dir()):
            continue
        if path.is_dir():
            continue
        if path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Source symlinks are not exported: {rel}")
        if len(rel.parts) == 1:
            allowed = rel.as_posix() in TOP
        else:
            allowed = rel.parts[0] in DIRECTORIES and (
                path.suffix in SUFFIXES or rel.parts[0] == "scripts" and not path.suffix
                or rel.as_posix() in SPECIAL)
        if not allowed:
            raise ValueError(f"Unreviewed file outside source allowlist: {rel}")
        found.append((rel.as_posix(), path))
    missing = TOP - {name for name, _ in found}
    if missing:
        raise ValueError("Required source files missing: " + ", ".join(sorted(missing)))
    return found


def audit():
    manifest = {}
    # Compare existing runtime credentials without printing or exporting them.
    runtime_secrets = []
    runtime = ROOT / "runtime"
    if runtime.exists():
        for path in runtime.rglob("*.token"):
            if path.is_file() and not path.is_symlink():
                value = path.read_bytes().strip()
                if len(value) >= 32:
                    runtime_secrets.append(value)
    files = source_files()
    snapshots = []
    license_text = ""
    for name, path in files:
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024 or b"\0" in data:
            raise ValueError(f"Binary or oversized source file: {name}")
        text = data.decode("utf-8")
        # Preserve an author's introductory README section: its SPDX marker can
        # live with the project documentation later in that file. An empty
        # README remains valid; other authored source needs a header marker.
        empty_readme = name == "README.md" and data == b""
        marker_region = text if name == "README.md" else text[:512]
        if name != "LICENSE" and not empty_readme and "SPDX-License-Identifier: Unlicense" not in marker_region:
            raise ValueError(f"Missing Unlicense SPDX marker: {name}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Credential/private key pattern in source: {name}")
        if any(secret in data for secret in runtime_secrets):
            raise ValueError(f"Runtime credential found in source: {name}")
        if str(Path.home()) + "/" in text:
            raise ValueError(f"Machine-specific home path in source: {name}")
        manifest[name] = hashlib.sha256(data).hexdigest()
        snapshots.append((name, data, bool(path.stat().st_mode & 0o111)))
        if name == "LICENSE":
            license_text = text
    if "This is free and unencumbered software released into the public domain." not in license_text:
        raise ValueError("LICENSE does not contain the standard Unlicense dedication")
    return snapshots, manifest


def add_file(archive, name, data, executable=False):
    info = tarfile.TarInfo("gpu-host-lab/" + name)
    info.size = len(data)
    info.mode = 0o755 if executable else 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def export(output, files, manifest):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, data, executable in files:
                    add_file(archive, name, data, executable)
                data = json.dumps({"license": "Unlicense", "distribution": "original text source only",
                                   "sha256": manifest}, indent=2, sort_keys=True).encode() + b"\n"
                add_file(archive, "SOURCE_MANIFEST.json", data)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "export"))
    parser.add_argument("--output", type=Path, default=ROOT / "dist/gpu-host-lab-source.tar.gz")
    args = parser.parse_args()
    try:
        files, manifest = audit()
        report = {"ok": True, "source_files": len(files), "license": "Unlicense",
                  "excluded": sorted(EXCLUDED | {"build*", "*.pyc"})}
        if args.command == "export":
            report.update(archive=str(args.output), sha256=export(args.output, files, manifest))
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
