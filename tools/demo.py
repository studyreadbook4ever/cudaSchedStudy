#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Run a reproducible multi-tenant demonstration on the actual local GPU."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import ctypes.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from gpuctl import Admin, Connection, HostError, Session


def job(response):
    return response.get("job", response)


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError("Demo condition did not become true")


@contextmanager
def host(args):
    if args.socket:
        if not args.admin_file:
            raise ValueError("--admin-file is required with --socket")
        with Admin.from_file(args.socket, args.admin_file) as admin:
            yield args.socket, admin, None
        return
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"Build the actual GPU daemon first: {binary}")
    with tempfile.TemporaryDirectory(prefix="gpu-demo-") as directory:
        path = Path(directory)
        socket_path = str(path / "socket" / "host.sock")
        admin_path = path / "private" / "admin.token"
        log_path = args.output.with_suffix(".log")
        trace_path = args.output.with_suffix(".trace.jsonl")
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [str(binary), "--socket", socket_path, "--admin-file", str(admin_path),
                 "--mode", args.mode, "--lanes", "4", "--sm-per-lane", "4", "--slots", "16",
                 "--max-elements", "262144", "--quantum", "1", "--batch-elements", "524288",
                 "--tick-ms", "2", "--trace", str(trace_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            admin = None
            try:
                def ready():
                    if process.poll() is not None:
                        raise RuntimeError(f"Actual GPU daemon exited; inspect {log_path}")
                    try:
                        if admin_path.is_file():
                            with Connection(socket_path) as connection:
                                return connection.ping()
                    except (OSError, ConnectionError, HostError):
                        pass
                    return False
                wait_until(ready, timeout=20)
                admin = Admin.from_file(socket_path, admin_path)
                yield socket_path, admin, trace_path
            finally:
                if admin is not None:
                    if process.poll() is None:
                        try:
                            admin.shutdown()
                        except (OSError, ConnectionError, HostError):
                            pass
                    admin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=3)


def run_demo(socket_path, admin):
    suffix = uuid.uuid4().hex[:8]
    names = [f"demo-{suffix}-{role}" for role in ("alpha", "beta", "returning")]
    tokens = [admin.provision(name, 3, weight)["token"] for name, weight in zip(names, (1, 3, 1))]
    sessions = []
    report = {"checks": {}}
    try:
        with Connection(socket_path) as connection:
            report["backend"] = connection.ping()["backend"]
        alpha = Session(socket_path, tokens[0])
        beta = Session(socket_path, tokens[1])
        sessions.extend((alpha, beta))
        report["participants"] = [alpha.hello, beta.hello]
        alpha_long = alpha.submit(65536, 100000, 3, 16)["job_id"]
        beta_long = beta.submit(65536, 100000, 3, 16)["job_id"]
        wait_until(lambda: alpha.stats()["completed_steps"] >= 10)
        low_start = alpha.stats()["scheduled_elements"]
        high_start = beta.stats()["scheduled_elements"]
        time.sleep(1.2)
        low = alpha.stats()["scheduled_elements"] - low_start
        high = beta.stats()["scheduled_elements"] - high_start
        report["weighted_scheduling"] = {"weights": [1, 3], "scheduled_operations": [low, high], "observed_ratio": high / low}
        report["checks"]["weighted_fairness"] = 2.0 < high / low < 4.2
        short = alpha.submit(4096, 16, 77, 3)["job_id"]
        another_short = alpha.submit(4096, 16, 78, 3)["job_id"]
        try:
            alpha.submit(1, 1, 0, 1)
            raise AssertionError("Tenant quota was not enforced")
        except HostError as error:
            report["quota_rejection"] = error.response
            report["checks"]["tenant_quota"] = error.response["error"] == "quota_exceeded"
        short_result = job(alpha.wait(short))
        another_result = job(alpha.wait(another_short))
        long_state = job(alpha.get(alpha_long))
        report["continuous_batching"] = {"newcomer": short_result, "another_newcomer": another_result, "existing_job": long_state, "multi_job_batches": alpha.stats()["global"]["multi_job_batches"]}
        report["checks"]["newcomer_finishes_while_existing_runs"] = short_result["state"] == "done" and long_state["state"] in {"queued", "running"}
        report["checks"]["multi_job_batch_executed"] = report["continuous_batching"]["multi_job_batches"] > 0
        alpha.cancel(alpha_long)
        report["cancelled"] = job(alpha.wait(alpha_long))
        report["checks"]["cancellation"] = report["cancelled"]["state"] == "cancelled" and report["cancelled"]["allocated_bytes"] == 0
        beta.close()
        def beta_departed():
            stats = next(item for item in admin.stats()["tenants"] if item.get("tenant", item.get("name")) == names[1])
            return stats if stats["active_sessions"] == 0 and stats["allocated_bytes"] == 0 else False
        report["disconnected"] = wait_until(beta_departed)
        report["checks"]["disconnect_reclaims_memory"] = report["disconnected"]["cancelled"] == 1
        old_lane = long_state["lane"]
        alpha.leave()
        def alpha_departed():
            return any(item.get("tenant", item.get("name")) == names[0] and item["lane"] == -1 for item in admin.stats()["tenants"])
        wait_until(alpha_departed)
        returning = Session(socket_path, tokens[2])
        sessions.append(returning)
        report["rejoined"] = returning.hello
        numeric = job(returning.wait(returning.submit(1025, 23, 42, 7)["job_id"]))
        report["checks"]["lane_reused"] = numeric["lane"] == old_lane
        libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        fmaf = libm.fmaf
        fmaf.argtypes = [ctypes.c_float] * 3
        fmaf.restype = ctypes.c_float
        f32 = lambda value: ctypes.c_float(value).value
        expected = []
        for index in list(range(8)) + [1024]:
            value = f32(f32(42 * f32(0.001)) + f32((index % 97) * f32(0.001)))
            for _ in range(23 * 7):
                value = fmaf(value, f32(0.9999), f32(f32(0.0001) * 42))
            expected.append(value)
        actual = numeric["samples"] + [numeric["last"]]
        max_error = max(abs(left - right) for left, right in zip(actual, expected))
        report["numerical"] = {"job": numeric, "cpu_reference": expected, "max_abs_error": max_error}
        report["checks"]["gpu_numerical_result"] = len(actual) == 9 and max_error <= 2e-6
        returning.leave()
        wait_until(lambda: admin.stats()["global"]["allocated_bytes"] == 0)
        report["final_stats"] = admin.stats()
        report["checks"]["all_job_slots_released"] = report["final_stats"]["global"]["allocated_bytes"] == 0
        report["passed"] = all(report["checks"].values())
        return report
    finally:
        for session in sessions:
            session.close()
        for name in names:
            try:
                admin.revoke(name)
            except (OSError, ConnectionError, HostError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(ROOT / "build" / "gpu-host"))
    parser.add_argument("--mode", choices=("green", "streams"), default="green")
    parser.add_argument("--socket", help="Use an already-running daemon instead of starting one")
    parser.add_argument("--admin-file")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "demo.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with host(args) as (socket_path, admin, trace):
            report = run_demo(socket_path, admin)
        if trace and trace.is_file():
            report["trace_file"] = str(trace)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "backend": report["backend"], "checks": report["checks"], "report": str(args.output)}, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, HostError, ValueError, TimeoutError, RuntimeError, AssertionError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
