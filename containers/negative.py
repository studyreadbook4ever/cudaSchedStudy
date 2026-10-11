#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Reject unsafe tenant configurations with temporary fixtures and no real secrets."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from manage import IMAGE, ROOT, docker, join, name, owned


def rejected(action, message):
    try:
        action()
    except RuntimeError as error:
        assert message in str(error), (message, str(error))
        return
    raise AssertionError("unsafe configuration was accepted")


def run_cases():
    cases = []
    def passed(case, action):
        action()
        cases.append({"case": case, "ok": True})
    with tempfile.TemporaryDirectory(prefix="gpu-container-negative-") as directory:
        temporary = Path(directory)
        socketdir = temporary / "socket"
        private = temporary / "private"
        socketdir.mkdir(mode=0o755)
        private.mkdir(mode=0o700)
        token = private / "fixture.token"
        token.write_text("fixture-only-not-a-real-credential\n")
        token.chmod(0o444)
        config = SimpleNamespace(tenant="negative_fixture", socket_dir=socketdir, token_file=token)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as endpoint:
            endpoint.bind(str(socketdir / "gpu.sock"))
            regular = socketdir / "ordinary-file"
            regular.write_text("fixture")
            passed("regular_file_in_shared_socket_directory", lambda: rejected(lambda: join(config), "sockets only"))
            regular.unlink()
            shared_token = socketdir / "credential.token"
            shared_token.write_text("fixture-only")
            shared_token.chmod(0o444)
            config.token_file = shared_token
            passed("credential_in_shared_socket_directory", lambda: rejected(lambda: join(config), "credentials must be outside"))
            shared_token.unlink()
            config.token_file = token
            token.chmod(0o600)
            passed("token_unreadable_by_mapped_nonroot_uid", lambda: rejected(lambda: join(config), "token file must be mode"))
            token.chmod(0o444)
            for invalid in ("UPPER", "../escape", "space user", "1first", "a" * 41):
                config.tenant = invalid
                passed("invalid_tenant_" + str(len(cases)), lambda: rejected(lambda: join(config), "tenant name must match"))
            config.tenant = "negative_fixture"
            supported = {"CgroupVersion": "2", "MemoryLimit": True, "SwapLimit": True,
                         "PidsLimit": True, "CpuCfsQuota": True}
            for feature in ("MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota", "CgroupVersion"):
                unsupported = dict(supported)
                unsupported[feature] = "1" if feature == "CgroupVersion" else False
                response = SimpleNamespace(stdout=json.dumps(unsupported))
                with patch("manage.docker", return_value=response) as mocked:
                    passed("unsupported_cgroup_" + feature, lambda: rejected(lambda: join(config), "working cgroup v2"))
                    assert mocked.call_count == 1, "unsupported host reached container creation"
    foreign = "foreignguard_" + uuid.uuid4().hex[:8]
    container = name(foreign)
    docker("run", "--detach", "--name", container, "--label", "gpu.shared.project=fixture:other-project",
           "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--user", "1000:1000", "--memory", "32m", "--memory-swap", "32m", "--pids-limit", "16",
           "--cpus", "0.2", "--entrypoint", "python", IMAGE, "-c", "import time; time.sleep(120)", capture=True)
    try:
        passed("foreign_project_owner_check", lambda: rejected(lambda: owned(foreign), "another project"))
        for operation in ("leave", "exec"):
            command = [str(ROOT / "scripts/container"), operation, foreign]
            if operation == "exec":
                command.extend(["python", "-c", "raise AssertionError('must never execute')"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert result.returncode == 1 and "another project" in result.stderr, result
            assert json.loads(docker("inspect", container, capture=True).stdout)[0]["State"]["Running"]
            cases.append({"case": "foreign_project_" + operation + "_refused_without_mutation", "ok": True})
    finally:
        docker("rm", "--force", container, capture=True)
    return {"ok": True, "cases": cases, "count": len(cases),
            "scope": "temporary fixture credentials, actual foreign-label Docker container, mocked unsupported-controller responses"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/container_negative.json")
    args = parser.parse_args()
    report = run_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ok": True, "negative_cases": report["count"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
