#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Measure real-GPU broker lifecycle costs with a fixed, repeatable workload.

Each scenario starts a fresh compiled broker. Capacity limitations are recorded
as observations, while wrong numerical results, leaked live work, and abnormal
daemon exits fail the benchmark. Never start this beside another GPU benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
from gpuctl import Admin, Connection, HostError, Session
from test_integration import job, reference


def distribution(values):
    ordered = sorted(values)
    def percentile(percent):
        offset = (len(ordered) - 1) * percent / 100
        low = math.floor(offset)
        high = math.ceil(offset)
        return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)
    return {"count": len(ordered), "mean_ms": statistics.fmean(ordered), "p50_ms": percentile(50),
            "p95_ms": percentile(95), "p99_ms": percentile(99), "max_ms": ordered[-1]} if ordered else {"count": 0}


def measured(operation):
    started = time.perf_counter_ns()
    result = operation()
    return result, (time.perf_counter_ns() - started) / 1e6


def process_metrics(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
    cpu_ms = (int(fields[11]) + int(fields[12])) * 1000 / os.sysconf("SC_CLK_TCK")
    status = Path(f"/proc/{pid}/status").read_text().splitlines()
    rss_kib = int(next(line for line in status if line.startswith("VmRSS:")).split()[1])
    return {"cpu_ms": cpu_ms, "rss_kib": rss_kib}


def numerical(result):
    result = job(result)
    if result["state"] != "done":
        raise AssertionError(f"Actual GPU job did not complete: {result['state']}")
    indices = list(range(min(8, result["elements"]))) + [result["elements"] - 1]
    expected = [reference(result["seed"], index, result["total_steps"], result["rounds"]) for index in indices]
    actual = result["samples"] + [result["last"]]
    if len(actual) != len(expected) or any(abs(left - right) > max(2e-7, abs(right) * 2e-6) for left, right in zip(actual, expected)):
        raise AssertionError("GPU result differs from the independent float32 CPU reference")
    return {"job_id": result["job_id"], "gpu_ms": result["gpu_ms"], "sm_mask": result["sm_mask"],
            "max_abs_error": max(abs(left - right) for left, right in zip(actual, expected))}


class Broker:
    def __init__(self, args, scenario):
        self.args = args
        self.scenario = scenario
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-lifecycle-")
        self.directory = Path(self.temp.name)
        self.socket = str(self.directory / "socket" / "gpu.sock")
        self.admin_file = self.directory / "private" / "admin.token"
        self.connections = []
        self.artifacts = args.output.with_suffix("") / scenario
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.log = (self.artifacts / "daemon.log").open("w")
        command = [str(args.binary), "--socket", self.socket, "--admin-file", str(self.admin_file),
                   "--mode", args.mode, "--lanes", "4", "--sm-per-lane", "4", "--slots", "16",
                   "--max-elements", "262144", "--quantum", "1", "--batch-elements", "524288", "--tick-ms", "2"]
        if args.trace:
            command.extend(("--trace", str(self.artifacts / "trace.jsonl")))
        start = time.monotonic()
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT)
        self.admin = None
        try:
            deadline = start + 20
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"GPU broker failed to start; inspect {self.artifacts / 'daemon.log'}")
                try:
                    if self.admin_file.is_file():
                        with Connection(self.socket) as connection:
                            self.backend = connection.ping()["backend"]
                        break
                except (OSError, ConnectionError):
                    pass
                time.sleep(0.01)
            else:
                raise TimeoutError("Actual GPU broker did not become ready")
            required = {"green": "green-contexts", "streams": "streams-shared"}[args.mode]
            if self.backend["mode"] != required:
                raise AssertionError("GPU benchmark cannot silently change backend mode")
            self.startup_ms = (time.monotonic() - start) * 1000
            self.admin = Admin.from_file(self.socket, self.admin_file)
            self.admin.stats()
        except BaseException:
            self.close()
            raise

    def token(self, name, quota=1, weight=1):
        return self.admin.provision(name, quota, weight)["token"]

    def join(self, token):
        session = Session(self.socket, token)
        self.connections.append(session)
        return session

    def quiesce(self, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stats = self.admin.stats()
            if stats["global"]["live_slots"] == 0 and stats["global"]["allocated_bytes"] == 0 and stats["global"]["inflight_batches"] == 0:
                return stats
            time.sleep(0.002)
        raise AssertionError("Broker left live GPU work or allocated memory after lifecycle cleanup")

    def close(self):
        for session in self.connections:
            session.close()
        if self.process.poll() is None:
            if self.admin is not None:
                try:
                    self.admin.shutdown()
                except (OSError, ConnectionError, HostError):
                    pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self.admin is not None:
            self.admin.close()
        self.log.close()
        self.temp.cleanup()
        if self.process.returncode != 0:
            raise AssertionError(f"GPU daemon exited with {self.process.returncode}")


@contextmanager
def broker(args, scenario):
    host = Broker(args, scenario)
    try:
        yield host
    finally:
        host.close()


def empty_sessions(host, args):
    token = host.token("session-benchmark")
    latencies = {"hello": [], "leave": [], "cycle": []}
    before = process_metrics(host.process.pid)
    for _ in range(args.cycles):
        started = time.perf_counter_ns()
        session, elapsed = measured(lambda: host.join(token))
        latencies["hello"].append(elapsed)
        _, elapsed = measured(session.leave)
        latencies["leave"].append(elapsed)
        latencies["cycle"].append((time.perf_counter_ns() - started) / 1e6)
    after = process_metrics(host.process.pid)
    return {"latencies": {name: distribution(values) for name, values in latencies.items()},
            "cycles_per_second": args.cycles / (sum(latencies["cycle"]) / 1000),
            "broker_cpu_ms": after["cpu_ms"] - before["cpu_ms"], "rss_before_kib": before["rss_kib"], "rss_after_kib": after["rss_kib"]}


def history_sessions(host, args):
    started = time.monotonic()
    for index in range(args.history_tenants):
        token = host.token(f"history{index}")
        session = host.join(token)
        for seed in range(args.history_jobs):
            job_id = session.submit(1, 100000, seed, 1)["job_id"]
            session.cancel(job_id)
            if job(session.wait(job_id, poll_interval=0.001))["state"] != "cancelled":
                raise AssertionError("History population failed to cancel its GPU job")
        session.leave()
    populated_ms = (time.monotonic() - started) * 1000
    result = empty_sessions(host, args)
    result.update(history_tenants=args.history_tenants, history_jobs_per_tenant=args.history_jobs,
                  retained_terminal_jobs=args.history_tenants * min(128, args.history_jobs), population_ms=populated_ms)
    return result


def completed_jobs(host, args):
    token = host.token("job-benchmark")
    wall = []
    proofs = []
    before = process_metrics(host.process.pid)
    for index in range(args.cycles):
        started = time.perf_counter_ns()
        session = host.join(token)
        job_id = session.submit(97, 2, index, 2)["job_id"]
        proofs.append(numerical(session.wait(job_id, poll_interval=0.001)))
        session.leave()
        wall.append((time.perf_counter_ns() - started) / 1e6)
    after = process_metrics(host.process.pid)
    return {"cycle": distribution(wall), "cycles_per_second": args.cycles / (sum(wall) / 1000),
            "broker_cpu_ms": after["cpu_ms"] - before["cpu_ms"], "numerical_jobs": len(proofs),
            "max_abs_error": max(proof["max_abs_error"] for proof in proofs), "gpu_ms": distribution([proof["gpu_ms"] for proof in proofs])}


def cancel_storm(host, args):
    token = host.token("cancel-benchmark", quota=16)
    session = host.join(token)
    ack = []
    terminal = []
    before = process_metrics(host.process.pid)
    progressed = 0
    for wave in range(args.storms):
        ids = [session.submit(65536, 100000, wave * 16 + index, 4)["job_id"] for index in range(16)]
        deadline = time.monotonic() + 10
        while job(session.get(ids[0]))["completed_steps"] == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("Cancellation benchmark never reached real GPU progress")
            time.sleep(0.001)
        progressed += 1
        started = time.perf_counter_ns()
        for job_id in ids:
            session.cancel(job_id)
        ack.append((time.perf_counter_ns() - started) / 1e6)
        for job_id in ids:
            if job(session.wait(job_id, poll_interval=0.001))["state"] != "cancelled":
                raise AssertionError("Cancellation storm left a non-cancelled long job")
        terminal.append((time.perf_counter_ns() - started) / 1e6)
        host.quiesce()
    after = process_metrics(host.process.pid)
    session.leave()
    return {"storms": args.storms, "jobs_per_storm": 16, "gpu_progressed_storms": progressed,
            "cancel_ack": distribution(ack), "cancel_terminal": distribution(terminal),
            "cancelled_jobs_per_second": args.storms * 16 / (sum(terminal) / 1000), "broker_cpu_ms": after["cpu_ms"] - before["cpu_ms"]}


def idle_participation(host, args):
    sessions = []
    capacity = 0
    for index in range(12):
        token = host.token(f"idle{index}")
        try:
            sessions.append(host.join(token))
        except HostError as error:
            if error.response["error"] != "capacity":
                raise
            capacity += 1
    idle = host.admin.stats()
    holders = sum(tenant["lane"] >= 0 and tenant["live_slots"] == 0 for tenant in idle["tenants"])
    session = sessions[-1]
    job_id = session.submit(97, 2, 42, 2)["job_id"]
    proof = numerical(session.wait(job_id, poll_interval=0.001))
    for session in sessions:
        session.leave()
    return {"attempted_idle_sessions": 12, "accepted_idle_sessions": len(sessions), "hello_capacity_denials": capacity,
            "idle_gpu_lane_holders": holders, "job_proof": proof}


def registry_churn(host, args):
    provision = []
    revoke = []
    forget = []
    supported = None
    completed = 0
    capacity_at = None
    before = process_metrics(host.process.pid)
    for index in range(args.registry_cycles):
        name = f"registry{index}"
        try:
            _, elapsed = measured(lambda: host.admin.provision(name, 1, 1))
        except HostError as error:
            if error.response["error"] != "capacity":
                raise
            capacity_at = index
            break
        provision.append(elapsed)
        _, elapsed = measured(lambda: host.admin.revoke(name))
        revoke.append(elapsed)
        if supported is not False:
            response, elapsed = measured(lambda: host.admin.command("FORGET", name, check=False))
            supported = response["ok"]
            if supported:
                forget.append(elapsed)
            elif response["error"] != "invalid_request":
                raise AssertionError(f"Drained revoked tenant could not be forgotten: {response['error']}")
        completed += 1
    after = process_metrics(host.process.pid)
    return {"requested_cycles": args.registry_cycles, "completed_cycles": completed, "capacity_at_cycle": capacity_at,
            "forget_supported": supported, "registered_tenants_after": len(host.admin.stats()["tenants"]),
            "provision": distribution(provision), "revoke": distribution(revoke), "forget": distribution(forget),
            "broker_cpu_ms": after["cpu_ms"] - before["cpu_ms"]}


SCENARIOS = {"empty": empty_sessions, "history": history_sessions, "jobs": completed_jobs,
             "cancel": cancel_storm, "idle": idle_participation, "registry": registry_churn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "gpu-host")
    parser.add_argument("--mode", choices=("green", "streams"), default="green")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--storms", type=int, default=10)
    parser.add_argument("--registry-cycles", type=int, default=200)
    parser.add_argument("--history-tenants", type=int, default=24)
    parser.add_argument("--history-jobs", type=int, default=128)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--trace", action="store_true", help="Include trace overhead; use consistently in comparisons")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "optimization" / "lifecycle.json")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    if min(args.cycles, args.storms, args.registry_cycles, args.history_tenants, args.history_jobs) <= 0 or args.history_tenants > 126 or args.history_jobs > 128:
        parser.error("Counts must be positive; history tenants<=126 and jobs<=128")
    if not args.binary.is_file():
        parser.error("Build the actual GPU broker before benchmarking")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(), "mode": args.mode,
              "client_sha256": hashlib.sha256((ROOT / "python" / "gpuctl.py").read_bytes()).hexdigest(),
              "trace_enabled": args.trace, "scenarios": {}}
    try:
        for name in args.scenarios:
            print(f"Measuring {name} on actual CUDA backend", flush=True)
            with broker(args, name) as host:
                data = SCENARIOS[name](host, args)
                final = host.quiesce()
                data.update(startup_ms=host.startup_ms, backend=host.backend, final_live_slots=final["global"]["live_slots"], final_allocated_bytes=final["global"]["allocated_bytes"])
            data["daemon_exit"] = host.process.returncode
            report["scenarios"][name] = data
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["completed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"completed": True, "report": str(args.output), "scenarios": list(report["scenarios"])}))
        return 0
    except (OSError, HostError, AssertionError, RuntimeError, TimeoutError) as error:
        report.update(completed=False, error=str(error))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"completed": False, "error": str(error), "report": str(args.output)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
