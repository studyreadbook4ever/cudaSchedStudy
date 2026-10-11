#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Container transaction tests with real file locks and a deterministic Docker adapter."""
import copy
import ast
import io
import json
import multiprocessing
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "containers"))
import manage
import audit


class DockerFixture:
    def __init__(self, socketdir, token):
        self.socketdir = socketdir
        self.token = token
        self.details = None
        self.calls = []
        self.timeouts = []
        self.failure = None

    def create(self, running=True, foreign=False):
        self.details = {"Id": "fixture-container-id", "State": {"Running": running},
                        "Config": {"Labels": {"gpu.shared.project": "foreign" if foreign else str(ROOT)}},
                        "Mounts": [{"Type": "bind", "Source": str(self.socketdir), "Destination": "/run/gpu", "RW": False},
                                   {"Type": "bind", "Source": str(self.token), "Destination": "/run/secrets/tenant.token", "RW": False}]}

    def __call__(self, *arguments, **kwargs):
        self.calls.append(arguments)
        self.timeouts.append((arguments[0], kwargs.get("timeout")))
        command = arguments[0]
        stdout, stderr, status = "", "", 0
        if command == "info":
            stdout = json.dumps({"CgroupVersion": "2", "MemoryLimit": True, "SwapLimit": True,
                                 "PidsLimit": True, "CpuCfsQuota": True})
        elif command == "inspect":
            if self.details is None:
                status, stderr = 1, "error: no such object: fixture"
            else:
                stdout = json.dumps([self.details])
        elif command == "run":
            self.create()
            if self.failure == "run_timeout":
                raise subprocess.TimeoutExpired("fixture-docker-run", .1)
            stdout = self.details["Id"] + "\n"
        elif command == "logs":
            if self.failure == "interrupt":
                raise KeyboardInterrupt()
            if self.failure == "logs_error":
                raise subprocess.TimeoutExpired("fixture-docker-logs", .1)
            if self.failure == "malformed":
                stdout = "invalid-json\n"
            elif self.failure == "unauthorized":
                stdout = json.dumps({"ok": False, "error": "unauthorized"}) + "\n"
            elif self.failure != "hello_timeout":
                stdout = json.dumps({"ok": True, "tenant": "fixture", "lane": -1}) + "\n"
        elif command == "exec":
            stdout = json.dumps({"ok": True, "tenant": {}, "global": {}}) + "\n"
            if self.failure == "health":
                status = 1
        elif command == "stop":
            self.details["State"]["Running"] = False
        elif command == "rm":
            self.details = None
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(arguments, status, stdout, stderr)

    def count(self, command):
        return sum(arguments[0] == command for arguments in self.calls)


class ContainerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpu-container-lifecycle-")
        self.base = Path(self.temporary.name)
        self.socketdir = self.base / "socket"
        self.socketdir.mkdir()
        self.private = self.base / "private"
        self.private.mkdir(mode=0o700)
        self.token = self.private / "fixture.token"
        self.token.write_text("fixture-only-not-a-real-credential\n")
        self.token.chmod(0o444)
        self.endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.endpoint.bind(str(self.socketdir / "gpu.sock"))
        self.args = SimpleNamespace(tenant="fixture", socket_dir=self.socketdir,
                                    token_file=self.token, join_timeout=.03)
        self.docker = DockerFixture(self.socketdir, self.token)
        self.patch_run = patch("manage.RUN", self.base / "run")
        self.patch_docker = patch("manage.docker", self.docker)
        self.patch_stdout = patch("sys.stdout", io.StringIO())
        self.patch_run.start()
        self.patch_docker.start()
        self.patch_stdout.start()

    def tearDown(self):
        self.patch_stdout.stop()
        self.patch_docker.stop()
        self.patch_run.stop()
        self.endpoint.close()
        self.temporary.cleanup()

    def test_repeated_join_reuses_one_live_hold(self):
        first = manage.join(self.args)
        second = manage.join(self.args)
        self.assertEqual(first, second)
        self.assertEqual(self.docker.count("run"), 1)
        self.assertEqual(self.docker.count("exec"), 1)
        self.assertEqual(self.docker.count("rm"), 0)

    def test_concurrent_join_is_serialized(self):
        errors = []
        def join():
            try:
                manage.join(self.args)
            except BaseException as error:
                errors.append(error)
        threads = [threading.Thread(target=join) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.docker.count("run"), 1)
        self.assertEqual(self.docker.count("exec"), 7)

    def test_join_failure_always_removes_created_container(self):
        for failure, expected in (("unauthorized", RuntimeError), ("malformed", ValueError),
                                  ("logs_error", subprocess.TimeoutExpired), ("interrupt", KeyboardInterrupt),
                                  ("run_timeout", subprocess.TimeoutExpired), ("hello_timeout", RuntimeError)):
            with self.subTest(failure=failure):
                self.docker.failure = failure
                self.docker.details = None
                before = self.docker.count("rm")
                with self.assertRaises(expected):
                    manage.join(self.args)
                self.assertIsNone(self.docker.details)
                self.assertEqual(self.docker.count("rm") - before, 1)

    def test_leave_is_idempotent(self):
        manage.join(self.args)
        manage.leave("fixture")
        manage.leave("fixture")
        self.assertIsNone(self.docker.details)
        self.assertEqual(self.docker.count("stop"), 1)
        self.assertEqual(self.docker.count("rm"), 1)

    def test_main_dispatches_leave_instead_of_calling_its_argument_parser(self):
        manage.join(self.args)
        with patch("sys.argv", ["container", "leave", "fixture"]), patch("manage.signal.signal"):
            manage.main()
        self.assertIsNone(self.docker.details)
        self.assertEqual(self.docker.count("stop"), 1)
        self.assertEqual(self.docker.count("rm"), 1)

    def test_inspect_distinguishes_missing_container_from_a_daemon_failure(self):
        for message in ("Error: No such object: fixture", "error: no such object: fixture",
                        "Error: No such container: fixture"):
            response = subprocess.CompletedProcess([], 1, "", message)
            with self.subTest(message=message), patch("manage.docker", return_value=response):
                self.assertIsNone(manage.inspect_container("fixture"))
        response = subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon")
        with patch("manage.docker", return_value=response), self.assertRaisesRegex(RuntimeError, "cannot inspect"):
            manage.inspect_container("fixture")

    def test_inspect_uses_normal_docker_timeout_during_container_teardown(self):
        response = subprocess.CompletedProcess([], 0, json.dumps([{"Id": "fixture"}]), "")
        with patch("manage.docker", return_value=response) as docker:
            manage.inspect_container("fixture")
            self.assertNotIn("timeout", docker.call_args.kwargs)

    def test_stopped_container_is_replaced(self):
        self.docker.create(running=False)
        manage.join(self.args)
        self.assertEqual(self.docker.count("rm"), 1)
        self.assertEqual(self.docker.count("run"), 1)
        self.assertTrue(self.docker.details["State"]["Running"])

    def test_foreign_container_is_never_changed(self):
        self.docker.create(foreign=True)
        original = copy.deepcopy(self.docker.details)
        for action in (lambda: manage.join(self.args), lambda: manage.leave("fixture"), lambda: manage.owned("fixture")):
            with self.assertRaisesRegex(RuntimeError, "another project"):
                action()
        self.assertEqual(self.docker.details, original)
        self.assertEqual(self.docker.count("run"), 0)
        self.assertEqual(self.docker.count("stop"), 0)
        self.assertEqual(self.docker.count("rm"), 0)

    def test_changed_mounts_do_not_cancel_running_work(self):
        self.docker.create()
        self.docker.details["Mounts"][0]["Source"] = "/fixture/other-socket"
        with self.assertRaisesRegex(RuntimeError, "mounts differ"):
            manage.join(self.args)
        self.assertEqual(self.docker.count("rm"), 0)
        self.assertEqual(self.docker.count("stop"), 0)

    def test_failed_health_probe_does_not_cancel_running_work(self):
        self.docker.create()
        self.docker.failure = "health"
        with self.assertRaisesRegex(RuntimeError, "health check"):
            manage.join(self.args)
        self.assertEqual(self.docker.count("rm"), 0)
        self.assertEqual(self.docker.count("stop"), 0)

    def test_join_rejects_invalid_timeout(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            self.args.join_timeout = timeout
            with self.assertRaisesRegex(RuntimeError, "finite and positive"):
                manage.join(self.args)
        self.assertEqual(self.docker.count("run"), 0)

    def test_authentication_deadline_does_not_limit_docker_layer_creation(self):
        manage.join(self.args)
        self.assertEqual([timeout for command, timeout in self.docker.timeouts if command == "run"], [30.0])

    def test_file_lock_serializes_independent_descriptors(self):
        entered = threading.Event()
        acquired = threading.Event()
        def competitor():
            entered.set()
            with manage.tenant_lock("fixture"):
                acquired.set()
        with manage.tenant_lock("fixture"):
            thread = threading.Thread(target=competitor)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(acquired.wait(.03))
        thread.join(2)
        self.assertTrue(acquired.is_set())
        self.assertEqual((self.base / "run/tenant-locks").stat().st_mode & 0o777, 0o700)

    def test_audit_write_probe_never_truncates_a_writable_credential(self):
        parsed = ast.parse(audit.PROBE)
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "writable")
        module = ast.Module(body=[function], type_ignores=[])
        scope = {"os": os}
        exec(compile(module, "fixture-audit-probe", "exec"), scope)
        original = self.token.read_bytes()
        self.token.chmod(0o600)
        result = scope["writable"](str(self.token), existing=True)
        self.assertFalse(result["blocked"])
        self.assertEqual(self.token.read_bytes(), original)
        probe_path = self.base / "root-probe"
        result = scope["writable"](str(probe_path))
        self.assertFalse(result["blocked"])
        self.assertFalse(probe_path.exists())

    def test_cli_sigterm_rolls_back_an_incomplete_join(self):
        context = multiprocessing.get_context("fork")
        reader, writer = context.Pipe(duplex=False)
        def child():
            fixture = DockerFixture(self.socketdir, self.token)
            def docker(*arguments, **kwargs):
                if arguments[0] == "logs":
                    writer.send({"stage": "created"})
                    time.sleep(30)
                return fixture(*arguments, **kwargs)
            arguments = ["container", "join", "fixture", "--socket-dir", str(self.socketdir),
                         "--token-file", str(self.token)]
            with patch("sys.argv", arguments), patch("manage.docker", docker):
                try:
                    manage.main()
                except KeyboardInterrupt:
                    writer.send({"stage": "rolled_back", "container_removed": fixture.details is None,
                                 "removals": fixture.count("rm")})
                finally:
                    writer.close()
        process = context.Process(target=child)
        process.start()
        writer.close()
        try:
            self.assertTrue(reader.poll(3), "join did not create a fixture container")
            self.assertEqual(reader.recv(), {"stage": "created"})
            os.kill(process.pid, signal.SIGTERM)
            self.assertTrue(reader.poll(3), "SIGTERM did not run join rollback")
            self.assertEqual(reader.recv(), {"stage": "rolled_back", "container_removed": True, "removals": 1})
            process.join(3)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            reader.close()


if __name__ == "__main__":
    unittest.main()
