# SPDX-License-Identifier: Unlicense
"""Exercise the public source archive workflow without CUDA or third-party tools."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_source_release", ROOT / "tools/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class SourceRelease(unittest.TestCase):
    def test_only_zero_byte_readme_is_exempt_from_spdx(self):
        with tempfile.TemporaryDirectory(prefix="gpu-source-readme-") as directory:
            temp = Path(directory)
            for name in release.TOP:
                content = (ROOT / "LICENSE").read_bytes() if name == "LICENSE" else b"# SPDX-License-Identifier: Unlicense\n"
                (temp / name).write_bytes(content)
            (temp / "README.md").write_bytes(b"")
            with mock.patch.object(release, "ROOT", temp):
                _, manifest = release.audit()
                self.assertEqual(manifest["README.md"], hashlib.sha256(b"").hexdigest())
                (temp / "README.md").write_text("A nonempty README needs SPDX.\n")
                with self.assertRaisesRegex(ValueError, "Missing Unlicense SPDX"):
                    release.audit()

    def test_readme_marker_can_follow_preserved_author_introduction(self):
        with tempfile.TemporaryDirectory(prefix="gpu-source-readme-section-") as directory:
            temp = Path(directory)
            for name in release.TOP:
                content = (ROOT / "LICENSE").read_bytes() if name == "LICENSE" else b"# SPDX-License-Identifier: Unlicense\n"
                (temp / name).write_bytes(content)
            introduction = b"# Theory\r\n" + b"Author-owned introduction.\r\n" * 40
            readme = introduction + b"# Architecture\r\n<!-- SPDX-License-Identifier: Unlicense -->\r\n"
            (temp / "README.md").write_bytes(readme)
            with mock.patch.object(release, "ROOT", temp):
                snapshot, manifest = release.audit()
                self.assertEqual(manifest["README.md"], hashlib.sha256(readme).hexdigest())
                archived_readme = next(data for name, data, _ in snapshot if name == "README.md")
                self.assertEqual(archived_readme[:len(introduction)], introduction)
                (temp / "CONTRIBUTING.md").write_bytes(b"Introduction\n" * 60 + b"# SPDX-License-Identifier: Unlicense\n")
                with self.assertRaisesRegex(ValueError, "Missing Unlicense SPDX marker: CONTRIBUTING.md"):
                    release.audit()

    def test_public_archive_is_complete_and_reusable(self):
        snapshot, manifest = release.audit()
        with tempfile.TemporaryDirectory(prefix="gpu-source-test-") as directory:
            temp = Path(directory)
            first = temp / "first.tar.gz"
            second = temp / "second.tar.gz"
            release.export(first, snapshot, manifest)
            release.export(second, snapshot, manifest)
            self.assertEqual(first.read_bytes(), second.read_bytes(), "Same source must export identically")
            with tarfile.open(first) as archive:
                names = {member.name for member in archive.getmembers()}
                for required in ("LICENSE", "CMakeLists.txt", "scripts/build.sh", "scripts/verify",
                                 "src/server.cpp", "src/gpu_backend.cu", "python/gpuctl.py",
                                 "docs/images/architecture.svg"):
                    self.assertIn("gpu-host-lab/" + required, names)
                self.assertFalse(any(part in {"runtime", "build", "artifacts", ".tools"}
                                     for name in names for part in Path(name).parts))
                self.assertTrue(all(not Path(name).is_absolute() and ".." not in Path(name).parts
                                    for name in names))
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(temp / "extracted", filter="data")
                else:
                    # This archive was generated locally above, with only regular
                    # files and checked relative paths; support older stdlib APIs.
                    archive.extractall(temp / "extracted")
            extracted = temp / "extracted/gpu-host-lab"
            proof = json.loads((extracted / "SOURCE_MANIFEST.json").read_text())
            self.assertEqual(len(names), len(proof["sha256"]) + 1)
            for name, digest in proof["sha256"].items():
                self.assertEqual(hashlib.sha256((extracted / name).read_bytes()).hexdigest(), digest)
            self.assertTrue((extracted / "scripts/build.sh").stat().st_mode & 0o111)
            command = [sys.executable, "tools/release.py"]
            for action in ("audit", "export"):
                completed = subprocess.run(command + [action], cwd=extracted, text=True,
                                           capture_output=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(first.read_bytes(), (extracted / "dist/gpu-host-lab-source.tar.gz").read_bytes())

    def test_export_preserves_audited_bytes_after_source_changes(self):
        # A release must not silently read newer source than its verified manifest.
        with tempfile.TemporaryDirectory(prefix="gpu-source-snapshot-") as directory:
            temp = Path(directory)
            original = b"# SPDX-License-Identifier: Unlicense\nprint('audited')\n"
            source = temp / "source.py"
            source.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            snapshot = [("source.py", source.read_bytes(), False)]
            source.write_text("changed after the audit\n")
            bundle = temp / "snapshot.tar.gz"
            release.export(bundle, snapshot, {"source.py": digest})
            with tarfile.open(bundle) as archive:
                self.assertEqual(archive.extractfile("gpu-host-lab/source.py").read(), original)
                manifest = json.loads(archive.extractfile("gpu-host-lab/SOURCE_MANIFEST.json").read())
                self.assertEqual(manifest["sha256"]["source.py"], digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
