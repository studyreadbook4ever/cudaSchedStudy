#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Small, dependency-free client for the GPU host's line protocol.

A Session owns a live connection: leave(), closing the socket, or exiting its
context cancels that connection's unfinished work. Keep it open while jobs run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import sys
import time
from typing import Any


class HostError(RuntimeError):
    def __init__(self, response: dict[str, Any]):
        self.response = response
        super().__init__(str(response.get("error", response)))


class _Stopped(Exception):
    """Interrupt a hold connection that is still authenticating."""


class Connection:
    def __init__(self, path: str, timeout: float = 10.0):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._buffer = bytearray()
        self._closed = False
        try:
            if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("connection timeout must be finite and positive, or None")
            self.socket.settimeout(timeout)
            self.socket.connect(path)
        except BaseException:
            self.close()
            raise

    def request(self, *fields: object, check: bool = True) -> dict[str, Any]:
        values = [str(field) for field in fields]
        if any(not value or any(char.isspace() for char in value) for value in values):
            raise ValueError("Protocol fields must be nonempty and contain no whitespace")
        return self.raw_request(" ".join(values), check=check)

    def raw_request(self, line: str | bytes, check: bool = True) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Connection is closed")
        encoded = line.encode("ascii") if isinstance(line, str) else line
        if b"\n" in encoded:
            raise ValueError("One request must contain exactly one command")
        try:
            self.socket.sendall(encoded + b"\n")
            while b"\n" not in self._buffer:
                data = self.socket.recv(65536)
                if not data:
                    raise ConnectionError("GPU host closed the connection")
                self._buffer.extend(data)
                if len(self._buffer) > 2 * 1024 * 1024:
                    raise ConnectionError("GPU host response exceeds client limit")
            raw, _, remaining = self._buffer.partition(b"\n")
            self._buffer = bytearray(remaining)
            response = json.loads(raw)
            if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
                raise ConnectionError("Invalid GPU host response")
        except BaseException:
            # A timed-out request can still receive a late response. Reusing
            # this stream would assign that response to the next command.
            self.close()
            raise
        if check and not response["ok"]:
            raise HostError(response)
        return response

    def ping(self) -> dict[str, Any]:
        return self.request("PING")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._buffer.clear()
            self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Admin(Connection):
    def __init__(self, path: str, token: str, timeout: float = 10.0):
        super().__init__(path, timeout)
        self.token = token.strip()

    @classmethod
    def from_file(cls, path: str, token_path: str | Path, timeout: float = 10.0):
        return cls(path, Path(token_path).read_text().strip(), timeout)

    def command(self, *fields: object, check: bool = True) -> dict[str, Any]:
        return self.request("ADMIN", self.token, *fields, check=check)

    def provision(self, name: str, quota_slots: int = 4, weight: int = 1):
        return self.command("PROVISION", name, quota_slots, weight)

    def stats(self):
        return self.command("STATS")

    def revoke(self, name: str):
        return self.command("REVOKE", name)

    def inspect(self, name: str):
        return self.command("INSPECT", name)

    def forget(self, name: str):
        return self.command("FORGET", name)

    def shutdown(self):
        return self.command("SHUTDOWN")


class Session(Connection):
    def __init__(self, path: str, token: str, timeout: float = 10.0):
        super().__init__(path, timeout)
        try:
            self.hello = self.request("HELLO", token.strip())
        except BaseException:
            self.close()
            raise

    def submit(self, elements: int, steps: int, seed: int = 1, rounds: int = 1):
        return self.request("SUBMIT", elements, steps, seed, rounds)

    def get(self, job_id: int):
        return self.request("GET", job_id)

    def jobs(self):
        return self.request("LIST")

    def heartbeat(self):
        return self.request("HEARTBEAT")

    def stats(self):
        response = self.request("STATS")
        # The wire response keeps tenant and global objects separate. Present
        # the caller's own counters directly while preserving the global view.
        if isinstance(response.get("tenant"), dict):
            return {"ok": response["ok"], **response["tenant"], "global": response["global"]}
        return response

    def cancel(self, job_id: int):
        return self.request("CANCEL", job_id)

    def wait(self, job_id: int, timeout: float = 30.0, poll_interval: float = 0.01):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("wait timeout must be finite and positive")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout:g} seconds")
            transport_timeout = self.socket.gettimeout()
            self.socket.settimeout(remaining if transport_timeout is None else min(remaining, transport_timeout))
            try:
                result = self.get(job_id)
            except socket.timeout as error:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout:g} seconds") from error
            finally:
                if not self._closed:
                    self.socket.settimeout(transport_timeout)
            job = result.get("job", result)
            if job.get("state") in {"done", "cancelled", "failed"}:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout:g} seconds")
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    def leave(self):
        try:
            return self.request("LEAVE")
        finally:
            self.close()


def _print(response: Any):
    print(json.dumps(response, ensure_ascii=False, sort_keys=True), flush=True)


def _token(args):
    if args.token_file:
        return Path(args.token_file).read_text().strip()
    token = args.token or os.environ.get("GPU_HOST_TOKEN")
    if not token:
        raise ValueError("Set --token, --token-file, or GPU_HOST_TOKEN")
    return token


def _shell(session: Session, timeout: float):
    _print(session.hello)
    print("submit ELEMENTS STEPS [SEED [ROUNDS]], get ID, wait ID, cancel ID, list, stats, leave", flush=True)
    interactive = sys.stdin.isatty()
    while True:
        try:
            if interactive:
                print("gpu> ", end="", flush=True)
            line = sys.stdin.readline()
            if not line:
                return
            fields = shlex.split(line)
            if not fields:
                continue
            command = fields[0].lower()
            if command in {"leave", "exit", "quit"}:
                _print(session.leave())
                return
            if command == "wait" and len(fields) == 2:
                _print(session.wait(int(fields[1]), timeout))
            elif command == "submit" and 3 <= len(fields) <= 5:
                values = [int(value) for value in fields[1:]]
                _print(session.submit(*values))
            elif command in {"get", "cancel"} and len(fields) == 2:
                _print(getattr(session, command)(int(fields[1])))
            elif command in {"list", "stats"} and len(fields) == 1:
                _print(session.jobs() if command == "list" else session.stats())
            else:
                print("Unknown command or wrong argument count", file=sys.stderr, flush=True)
        except (HostError, ValueError, TimeoutError) as error:
            _print(error.response if isinstance(error, HostError) else {"ok": False, "error": str(error)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=os.environ.get("GPU_HOST_SOCKET", "runtime/socket/gpu.sock"))
    parser.add_argument("--token")
    parser.add_argument("--token-file")
    parser.add_argument("--admin-file", default=os.environ.get("GPU_HOST_ADMIN_FILE", "runtime/private/admin.token"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--heartbeat", type=float, default=2.0, help="Seconds between authenticated health checks in hold mode")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ping")
    commands.add_parser("status", help="Tenant status (opens a temporary session)")
    commands.add_parser("heartbeat", help="Check authentication and broker health without returning job history")
    commands.add_parser("shell", help="Keep a session open and submit multiple jobs")
    commands.add_parser("hold", help="Join and remain connected until SIGTERM or Ctrl-C")
    run = commands.add_parser("run", help="Submit work, wait for GPU completion, and leave")
    run.add_argument("--elements", type=int, default=65536)
    run.add_argument("--steps", type=int, default=32)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--rounds", type=int, default=1)
    admin = commands.add_parser("admin")
    admin_commands = admin.add_subparsers(dest="admin_command", required=True)
    provision = admin_commands.add_parser("provision")
    provision.add_argument("name")
    provision.add_argument("--quota-slots", type=int, default=4)
    provision.add_argument("--weight", type=int, default=1)
    provision.add_argument("--token-out", "--output-token-file", dest="token_out", required=True, help="Write the tenant token to a new mode-0600 file; never print it")
    admin_commands.add_parser("stats")
    revoke = admin_commands.add_parser("revoke")
    revoke.add_argument("name")
    forget = admin_commands.add_parser("forget", help="Remove a revoked, idle tenant and its history")
    forget.add_argument("name")
    inspect = admin_commands.add_parser("inspect", help="Inspect one tenant including job history")
    inspect.add_argument("name")
    admin_commands.add_parser("shutdown")
    args = parser.parse_args(argv)
    stopping = False
    joined = False
    if args.command == "hold":
        # Install before HELLO: a container can receive SIGTERM immediately
        # after its session becomes visible to the administrator.
        def stop(*_):
            nonlocal stopping
            stopping = True
            if not joined:
                raise _Stopped()
    try:
        if args.command == "hold":
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be finite and positive")
        if not math.isfinite(args.heartbeat) or args.heartbeat <= 0:
            raise ValueError("--heartbeat must be finite and positive")
        if args.command == "ping":
            with Connection(args.socket, args.timeout) as connection:
                _print(connection.ping())
        elif args.command == "admin":
            with Admin.from_file(args.socket, args.admin_file, args.timeout) as connection:
                if args.admin_command == "provision":
                    # Reserve the secret file before creating server state, so
                    # an existing path cannot leave a tenant with a lost token.
                    fd = os.open(args.token_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    result = None
                    try:
                        with os.fdopen(fd, "w") as file:
                            result = connection.provision(args.name, args.quota_slots, args.weight)
                            file.write(result["token"] + "\n")
                    except BaseException:
                        Path(args.token_out).unlink(missing_ok=True)
                        if result is not None:
                            try:
                                connection.revoke(args.name)
                            except (OSError, ConnectionError, HostError):
                                pass
                        raise
                    result = {key: value for key, value in result.items() if key != "token"}
                    result["token_file"] = args.token_out
                    _print(result)
                elif args.admin_command in {"revoke", "forget", "inspect"}:
                    _print(getattr(connection, args.admin_command)(args.name))
                else:
                    _print(getattr(connection, args.admin_command)())
        else:
            with Session(args.socket, _token(args), args.timeout) as session:
                joined = True
                if args.command == "run":
                    submitted = session.submit(args.elements, args.steps, args.seed, args.rounds)
                    _print(submitted)
                    _print(session.wait(submitted["job_id"], args.timeout))
                elif args.command == "shell":
                    _shell(session, args.timeout)
                elif args.command == "hold":
                    session.socket.settimeout(min(args.timeout, args.heartbeat, 1.0))
                    _print(session.hello)
                    next_check = time.monotonic() + args.heartbeat
                    while not stopping:
                        time.sleep(min(0.1, args.heartbeat))
                        if not stopping and time.monotonic() >= next_check:
                            # A silent socket otherwise keeps a stale container
                            # alive after broker shutdown or credential revoke.
                            session.heartbeat()
                            next_check = time.monotonic() + args.heartbeat
                    _print(session.leave())
                elif args.command == "heartbeat":
                    _print(session.heartbeat())
                else:
                    _print(session.stats())
        return 0
    except _Stopped:
        return 0
    except (OSError, HostError, ValueError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
        _print(error.response if isinstance(error, HostError) else {"ok": False, "error": str(error)})
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
