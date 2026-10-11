#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Real Unix-socket client lifecycle tests that do not require CUDA or Docker."""
import json
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from gpuctl import Connection, HostError, Session


class Endpoint:
    def __init__(self, handler):
        self.directory = tempfile.TemporaryDirectory(prefix="gpu-client-protocol-")
        self.path = str(Path(self.directory.name) / "gpu.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen(1)
        self.listener.settimeout(3)
        self.errors = []

        def serve():
            try:
                with self.listener.accept()[0] as peer:
                    peer.settimeout(3)
                    handler(peer)
            except BaseException as error:
                self.errors.append(error)

        self.thread = threading.Thread(target=serve)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.thread.join(4)
        self.listener.close()
        self.directory.cleanup()
        if self.thread.is_alive():
            raise AssertionError("fixture socket thread did not exit")
        if self.errors:
            raise self.errors[0]


def line(peer):
    result = bytearray()
    while not result.endswith(b"\n"):
        value = peer.recv(1)
        if not value:
            return bytes(result)
        result.extend(value)
    return bytes(result)


def reply(peer, response):
    peer.sendall(json.dumps(response).encode() + b"\n")


def hello(peer):
    assert line(peer) == b"HELLO fixture-only\n"
    reply(peer, {"ok": True, "tenant": "fixture", "lane": -1})


class ClientProtocolTests(unittest.TestCase):
    def test_interrupted_transport_cannot_be_reused(self):
        class InterruptSocket:
            def __init__(self, stage):
                self.stage = stage
                self.closed = False
            def settimeout(self, timeout):
                pass
            def connect(self, path):
                pass
            def sendall(self, data):
                if self.stage == "send":
                    raise KeyboardInterrupt()
            def recv(self, size):
                raise KeyboardInterrupt()
            def close(self):
                self.closed = True
        for stage in ("send", "receive"):
            with self.subTest(stage=stage):
                transport = InterruptSocket(stage)
                with patch("gpuctl.socket.socket", return_value=transport):
                    client = Connection("fixture-only")
                    with self.assertRaises(KeyboardInterrupt):
                        client.ping()
                    self.assertTrue(transport.closed)
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        client.ping()

    def test_constructor_closes_after_timeout_configuration_failure(self):
        class InvalidSocket:
            def __init__(self):
                self.closed = False
            def settimeout(self, timeout):
                raise OverflowError("fixture socket timeout failure")
            def close(self):
                self.closed = True
        for timeout in (0, -1, float("nan"), float("inf"), 1):
            transport = InvalidSocket()
            with self.subTest(timeout=timeout), patch("gpuctl.socket.socket", return_value=transport):
                with self.assertRaises((ValueError, OverflowError)):
                    Connection("fixture-only", timeout)
                self.assertTrue(transport.closed)

    def test_fragmented_response(self):
        def handler(peer):
            self.assertEqual(line(peer), b"PING\n")
            peer.sendall(b'{"ok":')
            time.sleep(.01)
            peer.sendall(b'true,"ready":true}\n')
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint, Connection(endpoint.path) as client:
            self.assertTrue(client.ping()["ready"])

    def test_host_error_does_not_desynchronize_connection(self):
        def handler(peer):
            self.assertEqual(line(peer), b"BAD\n")
            reply(peer, {"ok": False, "error": "bad_command"})
            self.assertEqual(line(peer), b"PING\n")
            reply(peer, {"ok": True})
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint, Connection(endpoint.path) as client:
            with self.assertRaises(HostError):
                client.request("BAD")
            self.assertFalse(client._closed)
            self.assertTrue(client.ping()["ok"])

    def test_timeout_closes_stream_before_a_late_response_can_be_reused(self):
        def handler(peer):
            self.assertEqual(line(peer), b"PING\n")
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint, Connection(endpoint.path, .03) as client:
            with self.assertRaises(TimeoutError):
                client.ping()
            self.assertTrue(client._closed)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                client.ping()

    def test_invalid_response_closes_stream(self):
        for response in (b'{"ok":"yes"}\n', b'[]\n', b'not-json\n', b'\xff\n'):
            with self.subTest(response=response):
                def handler(peer):
                    self.assertEqual(line(peer), b"PING\n")
                    peer.sendall(response)
                    self.assertEqual(line(peer), b"")
                with Endpoint(handler) as endpoint, Connection(endpoint.path) as client:
                    with self.assertRaises((ValueError, ConnectionError)):
                        client.ping()
                    self.assertTrue(client._closed)

    def test_outbound_field_validation_preserves_connection(self):
        def handler(peer):
            self.assertEqual(line(peer), b"PING\n")
            reply(peer, {"ok": True})
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint, Connection(endpoint.path) as client:
            for fields in (("HAS SPACE",), ("",), ("HAS\nNEWLINE",)):
                with self.assertRaises(ValueError):
                    client.request(*fields)
            with self.assertRaises(ValueError):
                client.raw_request("PING\nPING")
            self.assertTrue(client.ping()["ok"])

    def test_wait_deadline_bounds_blocked_get(self):
        def handler(peer):
            hello(peer)
            self.assertEqual(line(peer), b"GET 12\n")
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint, Session(endpoint.path, "fixture-only", 2) as client:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                client.wait(12, timeout=.04)
            self.assertLess(time.monotonic() - started, .3)
            self.assertTrue(client._closed)

    def test_wait_rejects_invalid_timing(self):
        def handler(peer):
            hello(peer)
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint, Session(endpoint.path, "fixture-only") as client:
            for value in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    client.wait(1, timeout=value)
                with self.assertRaises(ValueError):
                    client.wait(1, poll_interval=value)

    def process(self, endpoint):
        return subprocess.Popen([sys.executable, str(ROOT / "python/gpuctl.py"),
                                 "--socket", endpoint.path, "--token", "fixture-only",
                                 "--heartbeat", ".03", "--timeout", ".5", "hold"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_hold_detects_revocation(self):
        def handler(peer):
            hello(peer)
            self.assertEqual(line(peer), b"HEARTBEAT\n")
            reply(peer, {"ok": False, "error": "revoked"})
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint:
            process = self.process(endpoint)
            try:
                output, errors = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 1, errors)
                self.assertEqual(json.loads(output.splitlines()[-1])["error"], "revoked")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_hold_sigterm_sends_leave(self):
        def handler(peer):
            hello(peer)
            while True:
                request = line(peer)
                if request == b"LEAVE\n":
                    reply(peer, {"ok": True})
                    break
                self.assertEqual(request, b"HEARTBEAT\n")
                reply(peer, {"ok": True, "tenant": "fixture", "lane": -1})
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint:
            process = self.process(endpoint)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=2))
                self.assertTrue(json.loads(process.stdout.readline())["ok"])
                process.send_signal(signal.SIGTERM)
                output, errors = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 0, errors)
                self.assertTrue(json.loads(output)["ok"])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_hold_sigterm_during_hello_closes_promptly(self):
        received = threading.Event()
        def handler(peer):
            self.assertEqual(line(peer), b"HELLO fixture-only\n")
            received.set()
            self.assertEqual(line(peer), b"")
        with Endpoint(handler) as endpoint:
            process = self.process(endpoint)
            try:
                self.assertTrue(received.wait(2))
                process.send_signal(signal.SIGTERM)
                output, errors = process.communicate(timeout=2)
                self.assertEqual(process.returncode, 0, errors)
                self.assertEqual(output, "")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
