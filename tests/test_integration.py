# SPDX-License-Identifier: Unlicense
"""Integration contracts against the compiled daemon and its real CUDA device.

No CPU implementation or mocked daemon is accepted. Select the deliberate
streams fallback with GPU_TEST_MODE=streams; green is tested by default.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from gpuctl import Admin, Connection, HostError, Session


def job(response):
    return response.get("job", response)


def f32(value):
    return ctypes.c_float(value).value


_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_fmaf = _libm.fmaf
_fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_fmaf.restype = ctypes.c_float


def reference(seed, index, steps, rounds):
    """Mirror CUDA float32 initialization and fused multiply-add rounding."""
    factor = seed % 97
    value = f32(f32(factor * f32(0.001)) + f32((index % 97) * f32(0.001)))
    bias = f32(f32(0.0001) * factor)
    coefficient = f32(0.9999)
    for _ in range(steps * rounds):
        value = _fmaf(value, coefficient, bias)
    return value


class RealGpuIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-host-test-")
        self.path = Path(self.temp.name)
        self.socket_path = str(self.path / "socket" / "host.sock")
        self.admin_path = self.path / "private" / "admin.token"
        self.trace_path = self.path / "trace.jsonl"
        self.mode = os.environ.get("GPU_TEST_MODE", "green")
        binary = Path(os.environ.get("GPU_HOST_BINARY", str(ROOT / "build" / "gpu-host"))).resolve()
        if not binary.is_file():
            self.temp.cleanup()
            self.fail(f"Actual compiled GPU daemon missing: {binary}; build it first")
        self.log_file = (self.path / "daemon.log").open("w+")
        # Make the cancellation kernel long enough to observe and cancel through
        # a socket round trip, even in streams mode using all 22 SMs.
        cancellation_test = self._testMethodName == "test_repeated_cancel_during_gpu_progress"
        starvation_test = self._testMethodName == "test_mixed_cost_jobs_do_not_starve"
        queued_history_test = self._testMethodName == "test_queued_cancellation_and_disconnect_keep_history_bounded"
        large_stats_test = self._testMethodName == "test_admin_stats_remains_bounded_with_large_terminal_history"
        slow_reader_test = self._testMethodName == "test_slow_reader_clients_are_reclaimed"
        quantum = "16" if cancellation_test else "1"
        budget = "512" if starvation_test else "16777216" if cancellation_test else "524288"
        admission_test = self._testMethodName in {
            "test_fifo_waiters_make_progress_with_more_tenants_than_lanes",
            "test_waiting_quota_disconnect_and_same_token_reconnect",
            "test_revoke_waiting_tenant_and_recycle_name",
            "test_unassigned_cancel_storm_and_eof_reclaim_queue",
        }
        self.lanes = 1 if starvation_test or admission_test else 4
        slots = "2" if starvation_test or admission_test else "16"
        maximum = "512" if starvation_test or admission_test else "262144"
        tick = "100" if queued_history_test or large_stats_test else "1" if starvation_test else "2"
        self.process = subprocess.Popen(
            [str(binary), "--socket", self.socket_path, "--admin-file", str(self.admin_path),
             "--lanes", str(self.lanes), "--sm-per-lane", "4", "--slots", slots, "--max-elements", maximum,
             "--quantum", quantum, "--batch-elements", budget, "--tick-ms", tick,
             "--max-clients", "2" if slow_reader_test else "128",
             "--trace", str(self.trace_path), "--mode", self.mode],
            cwd=ROOT, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.connections = []
        self.evidence = {}
        self.addCleanup(self.cleanup_daemon)
        self.backend = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_file.flush()
                self.log_file.seek(0)
                self.fail(f"Actual GPU daemon failed to start:\n{self.log_file.read()}")
            if self.admin_path.is_file():
                try:
                    with Connection(self.socket_path) as connection:
                        self.backend = connection.ping()["backend"]
                    break
                except (OSError, ConnectionError, HostError, KeyError):
                    pass
            time.sleep(0.025)
        self.assertIsNotNone(self.backend, "GPU daemon did not become ready")
        self.assertEqual(self.backend["mode"], {"green": "green-contexts", "streams": "streams-shared"}[self.mode], "Silent fallback is not permitted")
        self.assertGreater(self.backend["sm_count"], 0)
        self.assertTrue(self.backend["name"])
        if self.mode == "green":
            self.assertEqual(len(self.backend["sm_counts"]), self.lanes)
            self.assertTrue(all(count >= 4 for count in self.backend["sm_counts"]))
            self.assertLessEqual(sum(self.backend["sm_counts"]), self.backend["sm_count"])
        self.admin = Admin.from_file(self.socket_path, self.admin_path)
        self.connections.append(self.admin)

    def cleanup_daemon(self):
        snapshot = None
        try:
            if hasattr(self, "admin"):
                snapshot = self.admin.stats()
        except (OSError, ConnectionError, HostError):
            pass
        for connection in self.connections:
            connection.close()
        if self.process.poll() is None:
            try:
                with Admin.from_file(self.socket_path, self.admin_path) as admin:
                    admin.shutdown()
            except (OSError, ConnectionError, HostError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        self.log_file.close()
        artifact_root = Path(os.environ.get("GPU_TEST_ARTIFACT_DIR", str(ROOT / "artifacts" / f"integration-{self.mode}")))
        destination = artifact_root / self._testMethodName
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("daemon.log", "trace.jsonl"):
            source = self.path / filename
            if source.is_file():
                shutil.copyfile(source, destination / filename)
        (destination / "result.json").write_text(json.dumps({"backend": self.backend, "evidence": self.evidence, "final_stats": snapshot, "daemon_exit": self.process.returncode}, indent=2) + "\n")
        self.temp.cleanup()
        self.assertEqual(self.process.returncode, 0, f"The actual GPU daemon exited abnormally; inspect {destination / 'daemon.log'}")

    def provision(self, name, slots=4, weight=1):
        return self.admin.provision(name, slots, weight)["token"]

    def session(self, name, slots=4, weight=1, token=None):
        connection = Session(self.socket_path, token or self.provision(name, slots, weight))
        self.connections.append(connection)
        return connection

    def raw(self):
        connection = Connection(self.socket_path)
        self.connections.append(connection)
        return connection

    def tenant_stats(self, name):
        return next(item for item in self.admin.stats()["tenants"] if item.get("tenant", item.get("name")) == name)

    def eventually(self, predicate, timeout=5, message="Condition did not become true"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.005)
        self.fail(message)

    def assert_error(self, response, expected):
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error"], expected, response)

    def assert_numerical(self, result):
        result = job(result)
        self.assertEqual(result["state"], "done", result)
        self.assertEqual(result["completed_steps"], result["total_steps"])
        self.assertEqual(len(result["samples"]), min(8, result["elements"]))
        expected = [reference(result["seed"], index, result["total_steps"], result["rounds"]) for index in range(len(result["samples"]))]
        for actual, correct in zip(result["samples"], expected):
            self.assertTrue(math.isfinite(actual))
            self.assertAlmostEqual(actual, correct, delta=max(2e-7, abs(correct) * 2e-6))
        last = reference(result["seed"], result["elements"] - 1, result["total_steps"], result["rounds"])
        self.assertAlmostEqual(result["last"], last, delta=max(2e-7, abs(last) * 2e-6))
        self.assertAlmostEqual(result["first"], expected[0], delta=2e-7)
        self.assertAlmostEqual(result["sample_checksum"], sum(result["samples"]), delta=2e-6)
        self.assertEqual(result["allocated_bytes"], 0, "Terminal jobs must release their GPU slot")
        self.evidence.setdefault("numerical_checks", {})[str(result["job_id"])] = {"elements": result["elements"], "seed": result["seed"], "steps": result["total_steps"], "rounds": result["rounds"], "gpu_samples": result["samples"], "cpu_samples": expected, "gpu_last": result["last"], "cpu_last": last}

    def test_real_backend_and_private_files(self):
        self.assertEqual(stat.S_IMODE(self.admin_path.stat().st_mode), 0o600)
        self.assertEqual(len(self.admin_path.read_text().strip()), 64)
        self.assertGreater(self.backend["pool_bytes"], 0)
        self.assertGreater(self.backend["pinned_bytes"], 0)
        self.assertEqual(self.backend["slot_bytes"], self.backend["max_elements"] * 4)
        session = self.session("alice")
        self.assertEqual(session.hello["lane"], -1)
        self.assertEqual(session.stats()["global"]["resident_slots"], 0)
        print(f"\nActual CUDA backend: {self.backend['name']}, mode={self.mode}, SMs={self.backend['sm_count']}, lanes={self.backend['sm_counts']}", flush=True)

    def test_authentication_and_cross_tenant_ownership(self):
        anonymous = self.raw()
        self.assert_error(anonymous.request("SUBMIT", 1, 1, 1, 1, check=False), "not_authenticated")
        self.assert_error(anonymous.request("HELLO", "0" * 64, check=False), "unauthorized")
        self.assert_error(anonymous.request("ADMIN", "0" * 64, "STATS", check=False), "unauthorized")
        alice = self.session("alice")
        bob = self.session("bob")
        job_id = alice.submit(32768, 100000, 3, 4)["job_id"]
        self.assert_error(bob.request("GET", job_id, check=False), "not_found")
        self.assert_error(bob.request("CANCEL", job_id, check=False), "not_found")
        self.assert_error(bob.request("GET", 999999999, check=False), "not_found")
        self.assertEqual(bob.jobs()["jobs"], [])
        own_stats = bob.stats()
        self.assertNotIn("tenants", own_stats)
        self.assertNotIn("alice", json.dumps(own_stats))
        self.assert_error(alice.request("HELLO", self.provision("other"), check=False), "already_authenticated")
        alice.cancel(job_id)

    def test_authenticated_admin_survives_unauthenticated_timeout(self):
        self.assertTrue(self.admin.stats()["ok"])
        anonymous = self.raw()
        self.assertTrue(anonymous.ping()["ok"])
        time.sleep(5.3)
        self.assertTrue(self.admin.stats()["ok"], "An authenticated administrator was treated as an idle unauthenticated client")
        self.assertEqual(anonymous.socket.recv(1), b"", "Unauthenticated idle socket was not reclaimed")

    def test_numeric_reference_for_varied_shapes_and_seeds(self):
        session = self.session("numeric", slots=8)
        configurations = [(1, 1, 0, 1), (7, 3, 1, 7), (97, 19, 96, 3), (1025, 31, 97, 2),
                          (65536, 17, 4294967295, 5), (262144, 23, 42, 4)]
        submitted = [session.submit(*configuration)["job_id"] for configuration in configurations]
        for job_id in submitted:
            self.assert_numerical(session.wait(job_id, timeout=15))
        stats = session.stats()
        self.assertEqual(stats["live_slots"], 0)
        self.assertEqual(stats["allocated_bytes"], 0)
        self.assertEqual(stats["completed"], len(configurations))

    def test_quota_cancel_and_slot_reuse(self):
        session = self.session("quota", slots=1)
        first = session.submit(262144, 100000, 1, 16)["job_id"]
        stats = session.stats()
        self.assertEqual(stats["reserved_bytes"], self.backend["slot_bytes"])
        self.assertEqual(stats["live_slots"], 1)
        self.assert_error(session.request("SUBMIT", 1, 1, 2, 1, check=False), "quota_exceeded")
        session.cancel(first)
        cancelled = job(session.wait(first))
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertLess(cancelled["completed_steps"], cancelled["total_steps"])
        self.eventually(lambda: session.stats()["live_slots"] == 0)
        second = session.submit(1024, 7, 8, 3)["job_id"]
        self.assert_numerical(session.wait(second))
        self.assertEqual(session.stats()["allocated_bytes"], 0)

    def test_leave_disconnect_and_lane_reuse(self):
        token = self.provision("departing", slots=4)
        one = self.session("departing", token=token)
        two = self.session("departing", token=token)
        self.assertEqual(one.hello["lane"], -1)
        self.assertEqual(two.hello["lane"], -1)
        abandoned = one.submit(32768, 100000, 5, 4)["job_id"]
        old_lane = self.eventually(lambda: one.stats()["lane"] >= 0 and one.stats())["lane"]
        survivor = two.submit(1024, 3, 9, 1)["job_id"]
        one.leave()
        self.eventually(lambda: job(two.get(abandoned))["state"] == "cancelled")
        self.assert_numerical(two.wait(survivor))
        self.assertEqual(two.stats()["active_sessions"], 1)
        lost = two.submit(32768, 100000, 11, 4)["job_id"]
        two.close()
        self.eventually(lambda: self.tenant_stats("departing")["active_sessions"] == 0)
        self.eventually(lambda: self.tenant_stats("departing")["allocated_bytes"] == 0)
        replacement = self.session("replacement")
        self.assertEqual(replacement.hello["lane"], -1)
        replacement_id = replacement.submit(32768, 100000, 6, 4)["job_id"]
        self.assertEqual(replacement.stats()["lane"], old_lane)
        replacement.cancel(replacement_id)
        self.assertEqual(job(replacement.wait(replacement_id))["state"], "cancelled")
        renewed = self.session("departing", token=token)
        self.assertEqual(job(renewed.get(lost))["state"], "cancelled")

    def test_lane_capacity_and_dynamic_participation(self):
        sessions = [self.session(f"user{index}") for index in range(4)]
        pending = [session.submit(65536, 100000, index, 4)["job_id"] for index, session in enumerate(sessions)]
        self.assertEqual(len({session.stats()["lane"] for session in sessions}), 4)
        extra = self.session("extra", slots=1)
        self.assertEqual(extra.hello["lane"], -1)
        waiting_id = extra.submit(4096, 9, 23, 2)["job_id"]
        waiting = job(extra.get(waiting_id))
        self.assertEqual(waiting["state"], "queued")
        self.assertEqual(waiting["slot"], -1)
        self.assertEqual(waiting["allocated_bytes"], 0)
        self.assertEqual(extra.stats()["reserved_bytes"], self.backend["slot_bytes"])
        self.assertTrue(extra.stats()["waiting_for_lane"])
        old_lane = sessions[2].stats()["lane"]
        sessions[2].leave()
        self.assert_numerical(extra.wait(waiting_id))
        self.eventually(lambda: self.tenant_stats("user2")["lane"] == -1)
        self.assertEqual(extra.stats()["lane"], -1)
        assignment = [item for item in map(json.loads, self.trace_path.read_text().splitlines())
                      if item.get("event") == "lane_assigned" and item.get("tenant") == "extra"]
        self.assertEqual(assignment[0]["lane"], old_lane)
        for index, session in enumerate(sessions):
            if index != 2:
                session.cancel(pending[index])
                self.assertEqual(job(session.wait(pending[index]))["state"], "cancelled")

    def assert_reservations(self, snapshot):
        global_stats = snapshot["global"]
        tenants = snapshot["tenants"]
        self.assertEqual(global_stats["live_slots"], sum(item["live_slots"] for item in tenants))
        self.assertEqual(global_stats["resident_slots"], sum(item["live_slots"] for item in tenants if item["lane"] >= 0))
        self.assertEqual(global_stats["reserved_bytes"], global_stats["live_slots"] * self.backend["slot_bytes"])
        self.assertEqual(global_stats["allocated_bytes"], global_stats["resident_slots"] * self.backend["slot_bytes"])
        self.assertEqual(global_stats["waiting_tenants"], sum(item["waiting_for_lane"] for item in tenants))
        lanes = [item["lane"] for item in tenants if item["lane"] >= 0]
        self.assertEqual(len(lanes), len(set(lanes)), "Two resident tenants own the same GPU lane")
        for tenant in tenants:
            self.assertLessEqual(tenant["live_slots"], tenant["quota_slots"])
            self.assertEqual(tenant["reserved_bytes"], tenant["live_slots"] * self.backend["slot_bytes"])
            self.assertEqual(tenant["allocated_bytes"], 0 if tenant["lane"] < 0 else tenant["reserved_bytes"])

    def occupy_single_lane(self):
        blocker = self.session("blocker", slots=1)
        job_id = blocker.submit(512, 100000, 9, 16)["job_id"]
        self.eventually(lambda: job(blocker.get(job_id))["completed_steps"] > 0)
        self.assertEqual(blocker.stats()["lane"], 0)
        return blocker, job_id

    def test_idle_sessions_do_not_reserve_gpu_lanes(self):
        sessions = [self.session(f"idle{index}", slots=1) for index in range(12)]
        self.assertTrue(all(session.hello["lane"] == -1 for session in sessions))
        snapshot = self.admin.stats()
        self.assert_reservations(snapshot)
        self.assertEqual(snapshot["global"]["resident_slots"], 0)
        self.assertEqual(snapshot["global"]["waiting_tenants"], 0)
        for index, session in enumerate(sessions):
            result = session.wait(session.submit(97, 2, index, 2)["job_id"])
            self.assert_numerical(result)
            self.assertEqual(session.stats()["lane"], -1, "An idle authenticated connection retained its GPU lane")
            self.assertEqual(session.stats()["active_sessions"], 1)
        self.assert_reservations(self.admin.stats())
        self.evidence["idle_sessions"] = {"accepted": len(sessions), "idle_lane_holders": 0, "numerical_jobs": len(sessions)}

    def test_fifo_waiters_make_progress_with_more_tenants_than_lanes(self):
        blocker, blocker_id = self.occupy_single_lane()
        waiters = [self.session(f"waiting{index}", slots=1) for index in range(6)]
        pending = []
        for index, session in enumerate(waiters):
            response = session.submit(512, 2, index + 20, 1)
            self.assertTrue(response["waiting_for_lane"])
            pending.append(response["job_id"])
            result = job(session.get(pending[-1]))
            self.assertEqual((result["state"], result["slot"], result["completed_steps"]), ("queued", -1, 0))
        snapshot = self.admin.stats()
        self.assert_reservations(snapshot)
        self.assertEqual(snapshot["global"]["live_slots"], 7)
        self.assertEqual(snapshot["global"]["resident_slots"], 1)
        self.assertEqual(snapshot["global"]["waiting_tenants"], 6)
        blocker.cancel(blocker_id)
        self.assertEqual(job(blocker.wait(blocker_id))["state"], "cancelled")
        for session, job_id in zip(waiters, pending):
            self.assert_numerical(session.wait(job_id))
            self.assertEqual(session.stats()["lane"], -1)
        assignments = [item["tenant"] for item in map(json.loads, self.trace_path.read_text().splitlines())
                       if item.get("event") == "lane_assigned" and item["tenant"].startswith("waiting")]
        self.assertEqual(assignments, [f"waiting{index}" for index in range(6)], "Admission skipped the FIFO tenant queue")
        final = self.admin.stats()
        self.assert_reservations(final)
        self.assertEqual(final["global"]["waiting_tenants"], 0)
        self.assertEqual(final["global"]["live_slots"], 0)
        self.assertTrue(all(item["active_sessions"] == 1 and item["lane"] == -1 for item in final["tenants"]))
        self.evidence["fifo_admission"] = {"lanes": 1, "simultaneous_tenants": 7, "admission_order": assignments}

    def test_waiting_quota_disconnect_and_same_token_reconnect(self):
        blocker, blocker_id = self.occupy_single_lane()
        token = self.provision("pending", slots=2)
        owner = self.session("pending", token=token)
        observer = self.session("pending", token=token)
        ids = [owner.submit(512, 3, seed, 1)["job_id"] for seed in (31, 32)]
        self.assert_error(observer.request("SUBMIT", 1, 1, 0, 1, check=False), "quota_exceeded")
        stats = observer.stats()
        self.assertEqual((stats["lane"], stats["live_slots"], stats["allocated_bytes"]), (-1, 2, 0))
        self.assertEqual(stats["reserved_bytes"], 2 * self.backend["slot_bytes"])
        self.assertEqual(job(observer.cancel(ids[0]))["state"], "cancelled")
        owner.close()
        self.eventually(lambda: observer.stats()["live_slots"] == 0)
        for job_id in ids:
            result = job(observer.get(job_id))
            self.assertEqual(result["state"], "cancelled")
            self.assertEqual(result["completed_steps"], 0)
            self.assertEqual(result["allocated_bytes"], 0)
        self.assertFalse(observer.stats()["waiting_for_lane"])
        replacement = self.session("pending", token=token)
        self.assertEqual(replacement.hello["lane"], -1)
        next_id = replacement.submit(512, 3, 33, 2)["job_id"]
        self.assertEqual(job(observer.get(next_id))["slot"], -1)
        blocker.cancel(blocker_id)
        self.assertEqual(job(blocker.wait(blocker_id))["state"], "cancelled")
        self.assert_numerical(observer.wait(next_id))
        self.assertEqual(replacement.stats()["lane"], -1)
        self.assert_reservations(self.admin.stats())
        self.evidence["pending_reconnect"] = {"cancelled_without_gpu": ids, "resubmitted_numeric_job": next_id}

    def test_revoke_waiting_tenant_and_recycle_name(self):
        blocker, blocker_id = self.occupy_single_lane()
        token = self.provision("recycled", slots=2)
        owner = self.session("recycled", token=token)
        pending_id = owner.submit(512, 3, 41, 1)["job_id"]
        self.assertEqual(job(owner.get(pending_id))["slot"], -1)
        self.assert_error(self.admin.command("FORGET", "recycled", check=False), "tenant_busy")
        self.admin.revoke("recycled")
        snapshot = self.admin.command("INSPECT", "recycled")["tenant"]
        self.assertEqual((snapshot["enabled"], snapshot["live_slots"], snapshot["active_sessions"], snapshot["lane"]), (False, 0, 0, -1))
        self.assertEqual(snapshot["jobs"][0]["state"], "cancelled")
        self.assertEqual(self.admin.command("FORGET", "recycled")["forgotten"], "recycled")
        self.assert_error(self.admin.command("INSPECT", "recycled", check=False), "not_found")
        self.assert_error(self.raw().request("HELLO", token, check=False), "unauthorized")
        renewed_token = self.provision("recycled", slots=1)
        self.assertNotEqual(renewed_token, token)
        renewed = self.session("recycled", token=renewed_token)
        self.assertEqual(renewed.jobs()["jobs"], [])
        self.assert_error(renewed.request("GET", pending_id, check=False), "not_found")
        renewed_id = renewed.submit(512, 3, 42, 2)["job_id"]
        blocker.cancel(blocker_id)
        blocker.wait(blocker_id)
        self.assert_numerical(renewed.wait(renewed_id))
        self.assert_reservations(self.admin.stats())

    def test_registration_capacity_recycled_after_forget(self):
        self.assert_error(self.admin.command("FORGET", "missing", check=False), "not_found")
        first_token = None
        for index in range(200):
            token = self.provision("ephemeral", slots=1)
            if first_token is None:
                first_token = token
            connection = self.session("ephemeral", token=token)
            if index % 20 == 0:
                self.assert_numerical(connection.wait(connection.submit(97, 2, index, 2)["job_id"]))
            self.assert_error(self.admin.command("FORGET", "ephemeral", check=False), "tenant_busy")
            connection.leave()
            self.admin.revoke("ephemeral")
            self.assertTrue(self.admin.command("FORGET", "ephemeral")["ok"])
            snapshot = self.admin.stats()
            self.assertEqual(snapshot["tenants"], [])
            self.assert_reservations(snapshot)
        self.assert_error(self.raw().request("HELLO", first_token, check=False), "unauthorized")
        self.evidence["registration_churn"] = {"cycles": 200, "retained_tenants": 0, "numeric_jobs": 10}

    def test_unassigned_cancel_storm_and_eof_reclaim_queue(self):
        blocker, blocker_id = self.occupy_single_lane()
        token = self.provision("storm", slots=2)
        observer = self.session("storm", token=token)
        cancelled = 0
        for index in range(100):
            owner = self.session("storm", token=token)
            first = owner.submit(512, 5, index, 1)["job_id"]
            second = owner.submit(512, 5, index + 1, 1)["job_id"]
            self.assertEqual(job(observer.cancel(first))["state"], "cancelled")
            owner.close()
            self.eventually(lambda: observer.stats()["live_slots"] == 0)
            for job_id in (first, second):
                result = job(observer.get(job_id))
                self.assertEqual((result["state"], result["completed_steps"], result["slot"]), ("cancelled", 0, -1))
            self.assertFalse(observer.stats()["waiting_for_lane"])
            self.assertEqual(observer.stats()["reserved_bytes"], 0)
            self.assert_reservations(self.admin.stats())
            self.assertLessEqual(len(observer.jobs()["jobs"]), 128)
            cancelled += 2
        self.assertEqual(observer.stats()["cancelled"], cancelled)
        self.assertEqual(len(observer.jobs()["jobs"]), 128)
        blocker.cancel(blocker_id)
        blocker.wait(blocker_id)
        self.assert_numerical(observer.wait(observer.submit(512, 3, 55, 2)["job_id"]))
        self.assertEqual(self.admin.stats()["global"]["waiting_tenants"], 0)
        self.evidence["unassigned_cancel_storm"] = {"sessions": 100, "jobs_cancelled_without_gpu": cancelled, "history": 128}

    def test_repeated_session_churn_preserves_terminal_results(self):
        tokens = [self.provision(f"churn{index}", slots=2) for index in range(6)]
        observers = [self.session(f"churn{index}", token=token) for index, token in enumerate(tokens)]
        completed = 0
        cancelled = 0
        for index in range(180):
            tenant = index % len(tokens)
            owner = self.session(f"churn{tenant}", token=tokens[tenant])
            if index % 3 == 0:
                job_id = owner.submit(97 + index, 2, index, 2)["job_id"]
                self.assert_numerical(observers[tenant].wait(job_id))
                completed += 1
            else:
                job_id = owner.submit(65536, 100000, index, 16)["job_id"]
                if index % 2:
                    owner.leave()
                else:
                    owner.close()
                result = job(observers[tenant].wait(job_id))
                self.assertEqual(result["state"], "cancelled")
                self.assertLess(result["completed_steps"], result["total_steps"])
                cancelled += 1
            owner.close()
            self.eventually(lambda: observers[tenant].stats()["active_sessions"] == 1)
            self.eventually(lambda: observers[tenant].stats()["lane"] == -1)
            if index % 20 == 0:
                self.assert_reservations(self.admin.stats())
        final = self.admin.stats()
        self.assert_reservations(final)
        self.assertEqual(final["global"]["live_slots"], 0)
        self.assertEqual(sum(item["submitted"] for item in final["tenants"]), 180)
        self.assertEqual(sum(item["completed"] for item in final["tenants"]), completed)
        self.assertEqual(sum(item["cancelled"] for item in final["tenants"]), cancelled)
        self.assertTrue(all(item["active_sessions"] == 1 for item in final["tenants"]))
        self.evidence["session_churn"] = {"cycles": 180, "completed_numeric": completed, "cancelled": cancelled}

    def test_continuous_batching_admits_newcomers(self):
        session = self.session("batch", slots=4)
        long_id = session.submit(65536, 100000, 3, 4)["job_id"]
        self.eventually(lambda: job(session.get(long_id))["completed_steps"] >= 3)
        short_id = session.submit(4096, 16, 77, 2)["job_id"]
        other_id = session.submit(4096, 16, 78, 2)["job_id"]
        self.assert_numerical(session.wait(short_id, timeout=10))
        self.assert_numerical(session.wait(other_id, timeout=10))
        long_job = job(session.get(long_id))
        self.assertIn(long_job["state"], {"running", "queued"})
        self.assertLess(long_job["completed_steps"], long_job["total_steps"])
        self.assertGreater(session.stats()["global"]["multi_job_batches"], 0)
        session.cancel(long_id)
        self.assertEqual(job(session.wait(long_id))["state"], "cancelled")
        self.trace_path.read_text()  # Trace is produced by real batch completions.

    def test_weighted_cpu_scheduler(self):
        light = self.session("light", slots=2, weight=1)
        heavy = self.session("heavy", slots=2, weight=3)
        first = light.submit(65536, 100000, 5, 16)["job_id"]
        second = heavy.submit(65536, 100000, 5, 16)["job_id"]
        self.eventually(lambda: light.stats()["completed_steps"] >= 10, timeout=10)
        start_light = light.stats()["scheduled_elements"]
        start_heavy = heavy.stats()["scheduled_elements"]
        time.sleep(1.2)
        low = light.stats()["scheduled_elements"] - start_light
        high = heavy.stats()["scheduled_elements"] - start_heavy
        self.assertGreater(low, 0)
        ratio = high / low
        self.evidence["weighted_scheduling"] = {"light_operations": low, "heavy_operations": high, "ratio": ratio}
        print(f"\nWeighted GPU work (1:3): light={low}, heavy={high}, observed_ratio={ratio:.3f}", flush=True)
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 4.2)
        light.cancel(first)
        heavy.cancel(second)
        self.assertEqual(job(light.wait(first))["state"], "cancelled")
        self.assertEqual(job(heavy.wait(second))["state"], "cancelled")

    def test_invalid_commands_and_oversized_line(self):
        session = self.session("validation")
        commands = ["", "UNKNOWN", "SUBMIT", "SUBMIT 1 1 1 1 trailing", "SUBMIT -1 1 1 1",
                    "SUBMIT 0 1 1 1", "SUBMIT 262145 1 1 1", "SUBMIT 1 0 1 1", "SUBMIT 1 100001 1 1",
                    "SUBMIT 1 1 4294967296 1", "SUBMIT 1 1 -1 1", "SUBMIT 1 1 1 0", "SUBMIT 1 1 1 257",
                    "SUBMIT 1garbage 1 1 1", "GET -1", "CANCEL 1 extra"]
        for command in commands:
            with self.subTest(command=command):
                self.assert_error(session.raw_request(command, check=False), "invalid_request")
        oversized = self.raw()
        oversized.socket.sendall(b"X" * 70000 + b"\n")
        try:
            data = oversized.socket.recv(65536)
            if data:
                self.assertFalse(json.loads(data.split(b"\n", 1)[0])["ok"])
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.assertTrue(self.raw().ping()["ok"])
        self.assertEqual(session.stats()["live_slots"], 0)
        self.assert_numerical(session.wait(session.submit(97, 3, 8, 1)["job_id"]))

    def test_concurrent_gpu_stress_and_isolation(self):
        tokens = [self.provision(f"stress{index}", slots=3, weight=index + 1) for index in range(4)]
        failures = []
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def worker(index):
            try:
                with Session(self.socket_path, tokens[index]) as session:
                    barrier.wait(timeout=10)
                    for wave in range(5):
                        pending = [session.submit(1024 + index * 127 + offset, 7 + wave, 13 * index + wave + offset, 2 + index)["job_id"] for offset in range(3)]
                        for job_id in pending:
                            result = session.wait(job_id, timeout=20)
                            self.assert_numerical(result)
                            with lock:
                                results.append(job(result))
                    session.leave()
            except BaseException as error:
                with lock:
                    failures.append(error)
        workers = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=45)
        self.assertTrue(all(not thread.is_alive() for thread in workers), "Concurrent workers hung")
        if failures:
            raise failures[0]
        self.assertEqual(len(results), 60)
        self.assertEqual(len({result["job_id"] for result in results}), 60)
        self.eventually(lambda: self.admin.stats()["global"]["allocated_bytes"] == 0)
        for index in range(4):
            self.assertEqual(self.tenant_stats(f"stress{index}")["completed"], 15)
            self.eventually(lambda index=index: self.tenant_stats(f"stress{index}")["active_sessions"] == 0)

    def test_observed_sm_masks_match_partition_isolation(self):
        sessions = [self.session(f"partition{index}") for index in range(4)]
        pending = [session.submit(262144, 4, 11 + index, 16)["job_id"] for index, session in enumerate(sessions)]
        completed = [job(session.wait(job_id, timeout=15)) for session, job_id in zip(sessions, pending)]
        assigned_lanes = [result["lane"] for result in completed]
        self.assertEqual(len(set(assigned_lanes)), 4)
        self.assertTrue(all(lane >= 0 for lane in assigned_lanes))
        masks = [result["sm_mask"] for result in completed]
        self.evidence["observed_sm_masks"] = masks
        self.assertTrue(all(mask > 0 for mask in masks))
        if self.mode == "green":
            for index, mask in enumerate(masks):
                self.assertEqual(mask.bit_count(), self.backend["sm_counts"][assigned_lanes[index]])
                for other in masks[index + 1:]:
                    self.assertEqual(mask & other, 0, "Observed kernel SMs overlap across supposedly isolated tenants")
        print(f"\nObserved tenant SM masks ({self.backend['mode']}): {[hex(mask) for mask in masks]}", flush=True)

    def test_quota_shared_across_authenticated_connections(self):
        token = self.provision("shared", slots=2)
        first = self.session("shared", token=token)
        second = self.session("shared", token=token)
        first_id = first.submit(65536, 100000, 1, 4)["job_id"]
        second_id = second.submit(65536, 100000, 2, 4)["job_id"]
        self.assertEqual(first.stats()["live_slots"], 2)
        self.assertEqual(second.stats()["live_slots"], 2)
        self.assert_error(second.request("SUBMIT", 1, 1, 1, 1, check=False), "quota_exceeded")
        second.cancel(first_id)  # Ownership is tenant-wide, not connection-local.
        self.assertEqual(job(first.wait(first_id))["state"], "cancelled")
        first.cancel(second_id)
        self.assertEqual(job(second.wait(second_id))["state"], "cancelled")
        self.assert_numerical(first.wait(second.submit(1024, 9, 42, 2)["job_id"]))

    def test_pipelined_commands_are_fully_drained(self):
        connection = self.raw()
        connection.socket.settimeout(3)
        expected = 96
        connection.socket.sendall(b"PING\n" * expected)
        received = []
        buffer = bytearray()
        while len(received) < expected:
            data = connection.socket.recv(65536)
            self.assertTrue(data, f"Host closed after {len(received)} of {expected} commands")
            buffer.extend(data)
            while b"\n" in buffer:
                line, _, remaining = buffer.partition(b"\n")
                buffer = bytearray(remaining)
                received.append(json.loads(line))
        self.assertEqual(len(received), expected)
        self.evidence["pipeline_responses"] = len(received)
        self.assertTrue(all(item["ok"] and item["backend"]["mode"] == self.backend["mode"] for item in received))
        self.assertEqual(buffer, b"")
        self.assertTrue(connection.ping()["ok"])

    def test_half_close_preserves_complete_command_response(self):
        connection = self.raw()
        connection.socket.settimeout(3)
        connection.socket.sendall(b"PING\n")
        connection.socket.shutdown(socket.SHUT_WR)
        received = bytearray()
        while True:
            data = connection.socket.recv(65536)
            if not data:
                break
            received.extend(data)
        lines = received.splitlines()
        self.assertEqual(len(lines), 1, "A complete command was lost when the client half-closed")
        response = json.loads(lines[0])
        self.assertTrue(response["ok"])
        self.assertEqual(response["backend"]["mode"], self.backend["mode"])
        self.evidence["half_close_reply_count"] = len(lines)

    def test_mixed_cost_jobs_do_not_starve(self):
        session = self.session("mixed-cost", slots=2)
        expensive = session.submit(512, 10, 11, 256)["job_id"]
        cheap = session.submit(512, 100000, 23, 1)["job_id"]
        started = time.monotonic()
        def expensive_progress():
            status = job(session.get(expensive))
            return status if status["completed_steps"] >= 1 else False
        progress = self.eventually(expensive_progress, timeout=2, message="A cheap runnable job continually consumed credit and starved expensive GPU work")
        cheap_status = job(session.get(cheap))
        self.evidence["mixed_cost_progress"] = {"elapsed_seconds": time.monotonic() - started, "expensive_job": progress, "cheap_job": cheap_status}
        self.assertIn(cheap_status["state"], {"queued", "running"})
        self.assertLess(cheap_status["completed_steps"], cheap_status["total_steps"])
        session.cancel(expensive)
        session.cancel(cheap)
        self.assertEqual(job(session.wait(expensive))["state"], "cancelled")
        self.assertEqual(job(session.wait(cheap))["state"], "cancelled")
        self.assertEqual(session.stats()["allocated_bytes"], 0)

    def test_terminal_history_bounded_after_repeated_load(self):
        session = self.session("history", slots=1)
        ids = []
        for index in range(145):
            job_id = session.submit(1, 1, index, 1)["job_id"]
            ids.append(job_id)
            self.assertEqual(job(session.wait(job_id, poll_interval=0.001))["state"], "done")
        history = session.jobs()["jobs"]
        self.assertEqual(len(history), 128)
        self.assertEqual({item["job_id"] for item in history}, set(ids[-128:]))
        self.assert_error(session.request("GET", ids[0], check=False), "not_found")
        self.assertEqual(session.stats()["completed"], 145)
        self.assertEqual(session.stats()["allocated_bytes"], 0)

    def test_queued_cancellation_and_disconnect_keep_history_bounded(self):
        token = self.provision("queued-history", slots=1)
        owner = self.session("queued-history", token=token)
        observer = self.session("queued-history", token=token)
        retained = []
        for seed in range(128):
            job_id = owner.submit(1, 100000, seed, 1)["job_id"]
            owner.cancel(job_id)
            self.assertEqual(job(observer.wait(job_id, poll_interval=0.001))["state"], "cancelled")
            retained.append(job_id)
        self.assertEqual(len(observer.jobs()["jobs"]), 128)
        immediate = owner.submit(1, 100000, 128, 1)["job_id"]
        owner.cancel(immediate)
        self.assertEqual(job(observer.wait(immediate))["state"], "cancelled")
        retained.append(immediate)
        self.assertEqual(len(observer.jobs()["jobs"]), 128, "Immediate queued CANCEL left an extra terminal job outside the history bound")
        left = owner.submit(1, 100000, 129, 1)["job_id"]
        owner.leave()
        self.assertEqual(job(observer.wait(left))["state"], "cancelled")
        retained.append(left)
        self.assertEqual(len(observer.jobs()["jobs"]), 128, "Queued LEAVE cancellation exceeded the history bound")
        reconnect = self.session("queued-history", token=token)
        lost = reconnect.submit(1, 100000, 130, 1)["job_id"]
        reconnect.close()
        self.assertEqual(job(observer.wait(lost))["state"], "cancelled")
        retained.append(lost)
        history = observer.jobs()["jobs"]
        self.assertEqual(len(history), 128, "Queued EOF cancellation exceeded the history bound")
        self.assertEqual({entry["job_id"] for entry in history}, set(retained[-128:]))
        self.assertEqual(observer.stats()["allocated_bytes"], 0)
        self.evidence["queued_history"] = {"terminal_jobs": len(history), "submitted": len(retained), "routes": ["cancel", "leave", "eof"]}

    def test_terminal_history_retains_recent_completion_of_old_job(self):
        session = self.session("completion-order", slots=2)
        old = session.submit(4096, 100000, 23, 2)["job_id"]
        recent = []
        for seed in range(130):
            job_id = session.submit(1, 1, seed, 1)["job_id"]
            self.assert_numerical(session.wait(job_id, poll_interval=0.001))
            recent.append(job_id)
        self.assertIn(job(session.get(old))["state"], {"queued", "running"})
        session.cancel(old)
        terminal = job(session.wait(old))
        self.assertEqual(terminal["state"], "cancelled")
        history = session.jobs()["jobs"]
        self.assertEqual(len(history), 128)
        self.assertEqual({entry["job_id"] for entry in history}, {old, *recent[-127:]}, "History must keep recent terminal results even when their submission IDs are old")
        self.assert_error(session.request("GET", recent[0], check=False), "not_found")
        self.assertEqual(session.stats()["allocated_bytes"], 0)
        self.evidence["completion_history"] = {"old_job_id": old, "old_job_state": terminal["state"], "recent_terminal_ids": [entry["job_id"] for entry in history]}

    def test_randomized_multi_tenant_state_machine(self):
        random_source = random.Random(16602026)
        names = [f"model{index}" for index in range(8)]
        tokens = {name: self.provision(name, slots=3, weight=index + 1) for index, name in enumerate(names)}
        active = {name: [] for name in names}
        submitted = {}
        events = []
        closed_owners = set()
        cancelled_requests = set()

        def join(name):
            session = self.session(name, token=tokens[name])
            active[name].append(session)
            return session

        def add_job(name, session, elements, steps, seed, rounds):
            response = session.request("SUBMIT", elements, steps, seed, rounds, check=False)
            if response["ok"]:
                job_id = response["job_id"]
                submitted[job_id] = {"tenant": name, "owner": session.hello["session_id"], "elements": elements, "steps": steps, "seed": seed, "rounds": rounds}
                return job_id
            self.assertEqual(response["error"], "quota_exceeded", response)
            return None

        for name in names:
            join(name)
        second = join(names[0])
        for offset in range(3):
            owner = active[names[0]][offset % 2]
            self.assertIsNotNone(add_job(names[0], owner, 1024, 100000, offset, 3))
        self.assert_error(second.request("SUBMIT", 1, 1, 1, 1, check=False), "quota_exceeded")
        events.append({"action": "cross_session_quota", "tenant": names[0]})
        for job_id in list(submitted):
            second.cancel(job_id)
            self.assertEqual(job(second.wait(job_id))["state"], "cancelled")
            cancelled_requests.add(job_id)

        choices = ["submit"] * 5 + ["get", "cancel", "join", "leave", "eof", "foreign", "stats"] * 2
        for index in range(350):
            name = random_source.choice(names)
            action = random_source.choice(choices)
            if not active[name]:
                join(name)
            session = random_source.choice(active[name])
            own_ids = [job_id for job_id, metadata in submitted.items() if metadata["tenant"] == name]
            if action in {"get", "cancel"} and not own_ids:
                action = "submit"
            if action == "join" and len(active[name]) >= 3:
                action = "stats"
            event = {"index": index, "action": action, "tenant": name, "session_id": session.hello["session_id"]}
            if action == "submit":
                long_job = random_source.randrange(4) == 0
                values = (random_source.choice((1, 97, 513, 4096)), 100000 if long_job else random_source.randint(1, 12), random_source.randrange(2**32), random_source.randint(1, 5))
                event["job_id"] = add_job(name, session, *values)
                event["quota_rejected"] = event["job_id"] is None
            elif action == "get":
                job_id = random_source.choice(own_ids)
                result = job(session.get(job_id))
                self.assertEqual(result["tenant"], name)
                self.assertLessEqual(result["completed_steps"], result["total_steps"])
                if result["state"] == "done":
                    self.assert_numerical(result)
                event.update(job_id=job_id, state=result["state"])
            elif action == "cancel":
                job_id = random_source.choice(own_ids)
                result = job(session.cancel(job_id))
                self.assertEqual(result["tenant"], name)
                cancelled_requests.add(job_id)
                event.update(job_id=job_id, state=result["state"])
            elif action == "join":
                event["new_session_id"] = join(name).hello["session_id"]
            elif action in {"leave", "eof"}:
                closed_owners.add(session.hello["session_id"])
                if action == "leave":
                    session.leave()
                else:
                    session.close()
                active[name].remove(session)
            elif action == "foreign":
                foreign_ids = [job_id for job_id, metadata in submitted.items() if metadata["tenant"] != name]
                foreign_id = random_source.choice(foreign_ids) if foreign_ids else 2**63
                self.assert_error(session.request("GET", foreign_id, check=False), "not_found")
                self.assert_error(session.request("CANCEL", foreign_id, check=False), "not_found")
                event["job_id"] = foreign_id
            else:
                snapshot = session.stats()
                self.assertLessEqual(snapshot["live_slots"], snapshot["quota_slots"])
                self.assertEqual(snapshot["reserved_bytes"], snapshot["live_slots"] * self.backend["slot_bytes"])
                self.assertEqual(snapshot["allocated_bytes"], 0 if snapshot["lane"] < 0 else snapshot["reserved_bytes"])
            events.append(event)
            if index % 25 == 0:
                snapshot = self.admin.stats()
                self.assertEqual(snapshot["global"]["live_slots"], sum(item["live_slots"] for item in snapshot["tenants"]))
                self.assertEqual(snapshot["global"]["reserved_bytes"], snapshot["global"]["live_slots"] * self.backend["slot_bytes"])
                self.assertEqual(snapshot["global"]["allocated_bytes"], snapshot["global"]["resident_slots"] * self.backend["slot_bytes"])
                self.assertEqual(snapshot["global"]["allocated_bytes"], sum(item["allocated_bytes"] for item in snapshot["tenants"]))
                self.assertTrue(all(item["live_slots"] <= item["quota_slots"] for item in snapshot["tenants"]))

        completed_count = 0
        cancelled_count = 0
        for name in names:
            observer = active[name][0] if active[name] else join(name)
            for job_id, metadata in submitted.items():
                if metadata["tenant"] != name:
                    continue
                result = job(observer.get(job_id))
                if result["state"] not in {"done", "cancelled"}:
                    observer.cancel(job_id)
                    result = job(observer.wait(job_id))
                self.assertEqual(result["allocated_bytes"], 0)
                self.assertEqual(result["tenant"], name)
                if result["state"] == "done":
                    self.assert_numerical(result)
                    completed_count += 1
                else:
                    self.assertEqual(result["state"], "cancelled")
                    cancelled_count += 1
                if metadata["owner"] in closed_owners and metadata["steps"] == 100000:
                    self.assertEqual(result["state"], "cancelled", "A disconnected owner's long job survived cleanup")
            stats = observer.stats()
            expected = sum(metadata["tenant"] == name for metadata in submitted.values())
            self.assertEqual(stats["submitted"], expected)
            self.assertEqual(stats["completed"] + stats["cancelled"], expected)
            self.assertEqual(stats["live_slots"], 0)
            for session in active[name]:
                session.leave()
            active[name].clear()
        def quiescent():
            snapshot = self.admin.stats()
            if snapshot["global"]["live_slots"] == 0 and snapshot["global"]["inflight_batches"] == 0 and all(item["active_sessions"] == 0 and item["lane"] == -1 for item in snapshot["tenants"]):
                return snapshot
            return False
        final = self.eventually(quiescent)
        self.assertEqual(final["global"]["allocated_bytes"], 0)
        self.assertEqual(completed_count + cancelled_count, len(submitted))
        self.assertGreaterEqual(len(events), 350)
        self.assertTrue({"submit", "get", "cancel", "join", "leave", "eof", "foreign", "stats"} <= {event["action"] for event in events})
        self.evidence["randomized_model"] = {"seed": 16602026, "operations": len(events), "submitted_jobs": len(submitted), "completed": completed_count, "cancelled": cancelled_count, "cancel_requests": len(cancelled_requests), "closed_sessions": len(closed_owners), "events": events}

    def test_raw_authentication_and_partial_frame_boundaries(self):
        alice_token = self.provision("partial-alice", slots=2)
        bob_token = self.provision("partial-bob", slots=2)
        connection = self.raw()
        for command in ("GET 1", "CANCEL 1", "LIST", "STATS", "SUBMIT 1 1 1 1", "LEAVE"):
            self.assert_error(connection.raw_request(command, check=False), "not_authenticated")
        self.assert_error(connection.request("HELLO", alice_token[:-1], check=False), "unauthorized")
        self.assert_error(connection.request("ADMIN", alice_token, "SHUTDOWN", check=False), "unauthorized")
        self.assert_error(connection.request("ADMIN", alice_token, "PROVISION", "bypass", 1, 1, check=False), "unauthorized")
        self.assertFalse(any(item["name"] == "bypass" for item in self.admin.stats()["tenants"]))
        connection.socket.settimeout(0.05)
        connection.socket.sendall(b"PI")
        with self.assertRaises(socket.timeout):
            connection.socket.recv(1)
        connection.socket.settimeout(3)
        self.assertTrue(connection.raw_request(b"NG")["ok"])
        prefix = f"HELLO {alice_token[:17]}".encode()
        connection.socket.sendall(prefix)
        connection.socket.settimeout(0.05)
        with self.assertRaises(socket.timeout):
            connection.socket.recv(1)
        self.assertEqual(self.tenant_stats("partial-alice")["active_sessions"], 0)
        connection.socket.settimeout(3)
        authenticated = connection.raw_request(alice_token[17:])
        self.assertEqual(authenticated["tenant"], "partial-alice")
        self.assert_error(connection.request("HELLO", bob_token, check=False), "already_authenticated")
        own = connection.request("SUBMIT", 97, 3, 42, 2)["job_id"]
        bob = self.session("partial-bob", token=bob_token)
        self.assert_error(bob.request("GET", own, check=False), "not_found")
        self.assert_error(bob.request("CANCEL", own, check=False), "not_found")
        completed = self.eventually(lambda: result if (result := job(connection.request("GET", own)))["state"] == "done" else False)
        self.assert_numerical(completed)
        self.assertEqual(completed["tenant"], "partial-alice")
        incomplete = self.raw()
        incomplete.socket.sendall(b"HELLO ")
        incomplete.socket.shutdown(socket.SHUT_WR)
        buffer = bytearray()
        while True:
            data = incomplete.socket.recv(4096)
            if not data:
                break
            buffer.extend(data)
        replies = buffer.splitlines()
        self.assertEqual(len(replies), 1)
        self.assert_error(json.loads(replies[0]), "invalid_request")
        self.assertTrue(self.admin.stats()["ok"])
        self.evidence["partial_frames"] = {"ping_without_newline_unexecuted": True, "hello_without_newline_unexecuted": True, "incomplete_eof_replies": len(replies)}

    def test_eof_pipeline_drains_all_96_commands(self):
        connection = self.raw()
        connection.socket.settimeout(4)
        expected = 96
        connection.socket.sendall(b"PING\n" * expected)
        connection.socket.shutdown(socket.SHUT_WR)
        buffer = bytearray()
        while True:
            data = connection.socket.recv(65536)
            if not data:
                break
            buffer.extend(data)
        replies = [json.loads(line) for line in buffer.splitlines()]
        self.assertEqual(len(replies), expected, "Half-close discarded complete pipelined commands beyond a read quantum")
        self.assertTrue(all(reply["ok"] and reply["backend"]["mode"] == self.backend["mode"] for reply in replies))
        self.evidence["eof_pipeline"] = {"requests": expected, "replies": len(replies), "read_eof": True}

    def test_identity_capacity_rejects_129th_tenant(self):
        for index in range(128):
            self.provision(f"identity{index}", slots=1)
        self.assert_error(self.admin.command("PROVISION", "overflow", 1, 1, check=False), "capacity")
        self.assertEqual(len(self.admin.stats()["tenants"]), 128)
        self.assertTrue(self.raw().ping()["ok"])
        self.admin.revoke("identity0")
        self.admin.command("FORGET", "identity0")
        replacement = self.provision("replacement", slots=1)
        session = self.session("replacement", token=replacement)
        self.assert_numerical(session.wait(session.submit(17, 2, 18, 2)["job_id"]))
        self.assertEqual(len(self.admin.stats()["tenants"]), 128)
        self.evidence["identity_capacity"] = {"provisioned": 128, "overflow_rejected": True, "full_registry_slot_reclaimed": True}

    def test_admin_stats_remains_bounded_with_large_terminal_history(self):
        last_jobs = {}
        for index in range(24):
            name = f"history-volume{index}"
            session = self.session(name, slots=1)
            for seed in range(128):
                job_id = session.submit(1, 100000, seed, 1)["job_id"]
                session.cancel(job_id)
                self.assertEqual(job(session.wait(job_id, poll_interval=0.001))["state"], "cancelled")
            actual = session.submit(17, 3, index, 2)["job_id"]
            self.assert_numerical(session.wait(actual, timeout=5, poll_interval=0.001))
            last_jobs[name] = actual
            session.leave()
        snapshot = self.admin.stats()
        self.assertEqual(len(snapshot["tenants"]), 24)
        self.assertTrue(snapshot["jobs_omitted"])
        self.assertEqual(snapshot["global"]["allocated_bytes"], 0)
        self.assertLess(len(json.dumps(snapshot)), 65536, "Administrative aggregate stats grew with every retained job")
        for tenant in snapshot["tenants"]:
            self.assertEqual(tenant.get("jobs", []), [])
            self.assertEqual(tenant["submitted"], 129)
            self.assertEqual(tenant["completed"], 1)
            self.assertEqual(tenant["cancelled"], 128)
            self.assertEqual(tenant["allocated_bytes"], 0)
        inspected = []
        for name, last_id in last_jobs.items():
            details = self.admin.command("INSPECT", name)
            tenant = details["tenant"]
            self.assertEqual(tenant["name"], name)
            self.assertEqual(len(tenant["jobs"]), 128)
            recent = next(entry for entry in tenant["jobs"] if entry["job_id"] == last_id)
            self.assert_numerical(recent)
            inspected.append({"tenant": name, "retained_jobs": len(tenant["jobs"]), "latest_job": last_id})
        self.assert_error(self.admin.command("INSPECT", "absent", check=False), "not_found")
        unprivileged = self.raw()
        self.assert_error(unprivileged.request("ADMIN", "0" * 64, "INSPECT", next(iter(last_jobs)), check=False), "unauthorized")
        self.assertTrue(self.admin.stats()["ok"])
        self.evidence["large_admin_stats"] = {"tenants": 24, "queued_cancellations": 3072, "aggregate_response_bytes": len(json.dumps(snapshot)), "inspected": inspected}

    def test_slow_reader_clients_are_reclaimed(self):
        self.assertTrue(self.admin.stats()["ok"])
        slow = self.raw()
        slow.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        self.assertTrue(slow.ping()["ok"])
        with Connection(self.socket_path, timeout=0.5) as rejected:
            with self.assertRaises((ConnectionError, OSError)):
                rejected.ping()
        count = 1400
        slow.socket.sendall(b"PING\n" * count)
        slow.socket.shutdown(socket.SHUT_WR)
        started = time.monotonic()
        def replacement_accepted():
            try:
                with Connection(self.socket_path, timeout=0.2) as replacement:
                    return replacement.ping()["ok"]
            except (OSError, ConnectionError):
                return False
        self.eventually(replacement_accepted, timeout=4, message="Closing slow reader permanently occupied a bounded client slot")
        elapsed = time.monotonic() - started
        self.assertTrue(self.admin.stats()["ok"])
        self.evidence["slow_reader"] = {"max_clients": 2, "unread_ping_responses": count, "reclaimed_after_seconds": elapsed}

    def test_malformed_input_fuzz_1000_commands(self):
        session = self.session("fuzz", slots=1)
        random_source = random.Random(10001660)
        categories = {}
        fixed = [b"PING\r", b"PING\x00", b"\xffPING", b"SUBMIT 1 1 1 1\r", b"SUBMIT\t1 1 1 1",
                 b"SUBMIT 1 1 1 1 extra", b"GET", b"CANCEL 1 extra", b"SUBMIT 1 1 1", b"\x00HELLO", b"\xc3\xa9", b"UNKNOWN"]
        bad_by_field = [
            [b"0", b"-1", b"262145", b"18446744073709551615", b"18446744073709551616"],
            [b"0", b"-1", b"100001", b"18446744073709551616"],
            [b"-1", b"4294967296", b"18446744073709551616"],
            [b"0", b"-1", b"257", b"18446744073709551616"],
        ]
        universal_bad = [b"+1", b"1.0", b"1e2", b"0x10", b"1x", b"1\x00", b"1\r", b"\xff", b"9" * 80]
        for index in range(1000):
            category = random_source.randrange(4)
            if category == 0:
                command = random_source.choice(fixed)
            elif category == 1:
                field = random_source.randrange(4)
                values = [b"1", b"1", b"1", b"1"]
                values[field] = random_source.choice(bad_by_field[field] + universal_bad)
                command = b"SUBMIT " + b" ".join(values)
            elif category == 2:
                command = random_source.choice((b"GET ", b"CANCEL ")) + random_source.choice([b"0", b"-1", b"18446744073709551616"] + universal_bad)
            else:
                command = b" ".join([b"SUBMIT"] + [str(random_source.randrange(100)).encode() for _ in range(random_source.randint(8, 20))])
            categories[category] = categories.get(category, 0) + 1
            with self.subTest(seed=10001660, index=index, category=category):
                self.assert_error(session.raw_request(command, check=False), "invalid_request")
            if index % 100 == 0:
                self.assertTrue(self.admin.stats()["ok"])
        self.assertEqual(session.stats()["submitted"], 0)
        self.assertEqual(session.stats()["allocated_bytes"], 0)
        self.assert_numerical(session.wait(session.submit(4096, 11, 42, 3)["job_id"]))
        self.evidence["malformed_fuzz"] = {"seed": 10001660, "commands": 1000, "categories": categories}

    def test_repeated_cancel_during_gpu_progress(self):
        session = self.session("cancel-stress", slots=3)
        random_source = random.Random(1660)
        total = 0
        observed_inflight = False
        cancelled_inflight = False
        for wave in range(12):
            ids = [session.submit(262144, 100000, wave * 3 + offset, 256 if wave == 0 else 32)["job_id"] for offset in range(3)]
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                states = [job(session.get(job_id)) for job_id in ids]
                observed_inflight = observed_inflight or any(state["inflight"] for state in states)
                if any(state["inflight"] or state["completed_steps"] > 0 for state in states):
                    break
            order = ids[:]
            random_source.shuffle(order)
            inflight_ids = {state["job_id"] for state in states if state["inflight"]}
            order.sort(key=lambda job_id: job_id not in inflight_ids)
            for job_id in order:
                response = session.cancel(job_id)
                cancelled_inflight = cancelled_inflight or job(response)["inflight"]
            for job_id in ids:
                self.assertEqual(job(session.wait(job_id, timeout=15))["state"], "cancelled")
                total += 1
            self.assertEqual(session.stats()["allocated_bytes"], 0)
            self.assert_numerical(session.wait(session.submit(1025, 3, wave, 2)["job_id"]))
        self.assertEqual(session.stats()["cancelled"], total)
        self.evidence["cancellation"] = {"jobs": total, "observed_inflight": observed_inflight, "cancel_requested_while_inflight": cancelled_inflight}
        self.assertTrue(observed_inflight, "Test did not observe GPU work in flight")
        self.assertTrue(cancelled_inflight, "Test did not request cancellation while GPU work was in flight")
        print(f"\nRepeated cancellation: {total} jobs, observed_inflight={observed_inflight}, cancelled_inflight={cancelled_inflight}", flush=True)

    def test_revocation_and_cli_live_execution(self):
        token = self.provision("cli", slots=3)
        token_path = self.path / "private" / "cli.token"
        token_path.write_text(token + "\n")
        token_path.chmod(0o600)
        command = [sys.executable, str(ROOT / "python" / "gpuctl.py"), "--socket", self.socket_path,
                   "--token-file", str(token_path), "run", "--elements", "1025", "--steps", "11", "--seed", "23", "--rounds", "3"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        lines = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assert_numerical(lines[-1])
        session = self.session("cli", token=token)
        job_id = session.submit(32768, 100000, 2, 4)["job_id"]
        self.admin.revoke("cli")
        self.eventually(lambda: self.tenant_stats("cli")["allocated_bytes"] == 0)
        with self.assertRaises(HostError) as failure:
            Session(self.socket_path, token)
        self.assertEqual(failure.exception.response["error"], "unauthorized")
        self.assertEqual(self.tenant_stats("cli")["cancelled"], 1)

    def test_cli_provision_and_hold_session_lifecycle(self):
        token_path = self.path / "private" / "hold-user.token"
        base = [sys.executable, str(ROOT / "python" / "gpuctl.py"), "--socket", self.socket_path]
        provisioned = subprocess.run(base + ["--admin-file", str(self.admin_path), "admin", "provision", "holder",
                                             "--quota-slots", "2", "--weight", "2", "--token-out", str(token_path)],
                                     capture_output=True, text=True, timeout=10)
        self.assertEqual(provisioned.returncode, 0, provisioned.stdout + provisioned.stderr)
        public_result = json.loads(provisioned.stdout)
        self.assertNotIn("token", public_result)
        self.assertEqual(stat.S_IMODE(token_path.stat().st_mode), 0o600)
        token = token_path.read_text().strip()
        self.assertNotIn(token, provisioned.stdout + provisioned.stderr)
        repeated = subprocess.run(base + ["--admin-file", str(self.admin_path), "admin", "provision", "accidental",
                                          "--token-out", str(token_path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(repeated.returncode, 1)
        self.assertFalse(any(item["name"] == "accidental" for item in self.admin.stats()["tenants"]), "A failed secret-file reservation created an unreachable tenant")
        self.assertEqual(token_path.read_text().strip(), token)
        holder = subprocess.Popen(base + ["--token-file", str(token_path), "hold"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.eventually(lambda: self.tenant_stats("holder")["active_sessions"] == 1)
            holder.terminate()
            output, errors = holder.communicate(timeout=10)
            self.assertEqual(holder.returncode, 0, output + errors)
            self.assertNotIn(token, output + errors)
            replies = [json.loads(line) for line in output.splitlines()]
            self.assertEqual(replies[0]["tenant"], "holder")
            self.assertTrue(replies[-1]["bye"])
            self.eventually(lambda: self.tenant_stats("holder")["active_sessions"] == 0)
            self.eventually(lambda: self.tenant_stats("holder")["lane"] == -1)
            revoked_holder = subprocess.Popen(base + ["--token-file", str(token_path), "--heartbeat", "0.1", "hold"],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.eventually(lambda: self.tenant_stats("holder")["active_sessions"] == 1)
                self.admin.revoke("holder")
                output, errors = revoked_holder.communicate(timeout=3)
                self.assertEqual(revoked_holder.returncode, 1, output + errors)
                self.assertNotIn(token, output + errors)
                self.eventually(lambda: self.tenant_stats("holder")["active_sessions"] == 0)
            finally:
                if revoked_holder.poll() is None:
                    revoked_holder.kill()
                    revoked_holder.communicate(timeout=3)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.communicate(timeout=3)

    def test_startup_rejects_trace_aliases_without_destroying_files(self):
        self.admin.shutdown()
        self.assertEqual(self.process.wait(timeout=10), 0)
        binary = Path(os.environ.get("GPU_HOST_BINARY", str(ROOT / "build" / "gpu-host"))).resolve()
        records = []
        cases = ("admin_path", "admin_hardlink", "admin_symlink", "socket_absent", "socket_existing")
        for case in cases:
            with self.subTest(case=case):
                directory = self.path / case
                private = directory / "private"
                public = directory / "socket"
                private.mkdir(parents=True)
                public.mkdir()
                admin_path = private / "admin.token"
                admin_bytes = self.admin_path.read_bytes()
                admin_path.write_bytes(admin_bytes)
                admin_path.chmod(0o600)
                socket_path = public / "gpu.sock"
                if case == "admin_path":
                    trace_path = admin_path
                elif case == "admin_hardlink":
                    trace_path = directory / "trace.jsonl"
                    os.link(admin_path, trace_path)
                elif case == "admin_symlink":
                    trace_path = directory / "trace.jsonl"
                    trace_path.symlink_to(admin_path)
                else:
                    trace_path = socket_path
                socket_bytes = b"An existing socket pathname must survive rejected startup.\n"
                if case == "socket_existing":
                    socket_path.write_bytes(socket_bytes)
                command = [str(binary), "--socket", str(socket_path), "--admin-file", str(admin_path),
                           "--trace", str(trace_path), "--mode", self.mode, "--lanes", "1", "--sm-per-lane", "4",
                           "--slots", "1", "--max-elements", "512"]
                rejected = subprocess.run(command, capture_output=True, text=True, timeout=8)
                self.assertEqual(rejected.returncode, 1, "Conflicting trace paths must fail startup explicitly")
                self.assertIn("trace", rejected.stderr.lower())
                self.assertNotIn("\"event\":\"ready\"", rejected.stdout)
                self.assertEqual(admin_path.read_bytes(), admin_bytes, "Rejected startup destroyed the administrator credential")
                self.assertEqual(stat.S_IMODE(admin_path.stat().st_mode), 0o600)
                if case == "socket_existing":
                    self.assertEqual(socket_path.read_bytes(), socket_bytes, "Rejected startup truncated an existing socket pathname")
                else:
                    self.assertFalse(socket_path.exists(), "Rejected startup created or removed an unrelated socket pathname")
                records.append({"case": case, "exit": rejected.returncode, "admin_preserved": True, "socket_preserved": True})
        self.evidence["startup_trace_conflicts"] = records

    def shutdown_output_pipeline(self, count=12):
        for index in range(128):
            self.provision(f"shutdown-identity{index}", slots=1)
        response_bytes = len(json.dumps(self.admin.stats(), separators=(",", ":")).encode()) + 1
        self.assertGreater(response_bytes * count, 300000, "Regression must exceed the UNIX socket send buffer")
        self.assertLess(response_bytes * count, 1048576, "Use the output flush path rather than the output size limit")
        connection = self.raw()
        connection.socket.settimeout(5)
        stats_command = f"ADMIN {self.admin.token} STATS\n".encode()
        shutdown_command = f"ADMIN {self.admin.token} SHUTDOWN\n".encode()
        connection.socket.sendall(stats_command * count + shutdown_command)
        return connection, response_bytes * count

    def test_shutdown_flushes_large_pipelined_responses(self):
        connection, estimated_bytes = self.shutdown_output_pipeline()
        time.sleep(0.15)
        received = bytearray()
        while True:
            chunk = connection.socket.recv(65536)
            if not chunk:
                break
            received.extend(chunk)
            self.assertLess(len(received), 2 * 1048576)
        lines = bytes(received).splitlines()
        self.evidence["shutdown_output_flush"] = {"responses": len(lines), "received_bytes": len(received), "estimated_stats_bytes": estimated_bytes, "reader_delay_ms": 150}
        self.assertTrue(received.endswith(b"\n"), "Shutdown truncated a JSON response")
        self.assertEqual(len(lines), 13, "Shutdown lost previously accepted responses or its final acknowledgement")
        replies = [json.loads(line) for line in lines]
        for response in replies[:-1]:
            self.assertTrue(response["ok"])
            self.assertEqual(len(response["tenants"]), 128)
        self.assertEqual(replies[-1], {"ok": True, "draining": True})
        self.assertEqual(self.process.wait(timeout=4), 0)
        self.evidence["shutdown_output_flush"] = {"responses": len(replies), "received_bytes": len(received), "estimated_stats_bytes": estimated_bytes, "reader_delay_ms": 150}

    def test_shutdown_deadline_reclaims_unread_output(self):
        connection, estimated_bytes = self.shutdown_output_pipeline()
        started = time.monotonic()
        self.assertEqual(self.process.wait(timeout=4), 0, "An unread peer prevented the actual GPU daemon from shutting down")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 3.5)
        self.assertFalse(Path(self.socket_path).exists())
        connection.close()
        self.evidence["shutdown_unread_peer"] = {"estimated_stats_bytes": estimated_bytes, "exit_elapsed_seconds": elapsed, "read_response_bytes": 0}

    def test_graceful_shutdown_with_active_gpu_work(self):
        sessions = [self.session(f"shutdown{index}") for index in range(4)]
        for index, session in enumerate(sessions):
            session.submit(65536, 100000, index, 4)
        self.eventually(lambda: self.admin.stats()["global"]["scheduled_elements"] > 0)
        self.assertTrue(self.admin.shutdown()["ok"])
        self.assertEqual(self.process.wait(timeout=10), 0)
        self.assertFalse(Path(self.socket_path).exists(), "Shutdown must unlink its service socket")


if __name__ == "__main__":
    unittest.main(verbosity=2)
