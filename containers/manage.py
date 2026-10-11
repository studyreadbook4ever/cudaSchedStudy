#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Project-local rootless Docker and restricted tenant lifecycle commands."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser().resolve()
TOOLS = CACHE / "gpu-host-lab/docker"
RUN = TOOLS / "run"
SOCKET = RUN / "docker.sock"
IMAGE = "gpu-shared-tenant:local"
LABEL = "gpu.shared.project=" + str(ROOT)
VERSION = "29.9.0"
ARCHIVES = {
    f"docker-{VERSION}.tgz": "33e1ab8b63d14bca449f7a3d30d7f6aa669daa544e60a86b17fec87f61afe2b4",
    f"docker-rootless-extras-{VERSION}.tgz": "6b4a2e085a1e4ade5ba3de99fd770a0a9b53c719e6bf6b1632242bf4b068adb6",
}


def docker(*arguments, capture=False, check=True, timeout=30):
    executable = TOOLS / "bin/docker"
    if not executable.exists():
        raise RuntimeError("run scripts/container bootstrap and scripts/container daemon-start first")
    command = [str(executable), "--config", str(TOOLS / "config"), "--host", "unix://" + str(SOCKET), *arguments]
    return subprocess.run(command, check=check, text=True, capture_output=capture, timeout=timeout)


def ready():
    try:
        return SOCKET.exists() and docker("info", capture=True, check=False, timeout=2).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def launcher_alive(pid):
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
        return status.rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def bootstrap():
    if os.uname().machine != "x86_64":
        raise RuntimeError("this pinned static Docker installation supports x86_64 only")
    for directory in (TOOLS / "bin", TOOLS / "downloads", TOOLS / "config", TOOLS / "data", RUN):
        directory.mkdir(parents=True, exist_ok=True)
    RUN.chmod(0o700)
    for filename, expected in ARCHIVES.items():
        destination = TOOLS / "downloads" / filename
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            url = "https://download.docker.com/linux/static/stable/x86_64/" + filename
            print("Downloading", url, flush=True)
            urllib.request.urlretrieve(url, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise RuntimeError("SHA256 mismatch for " + filename)
        with tarfile.open(destination, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                target = TOOLS / "bin" / name
                descriptor, temporary = tempfile.mkstemp(prefix="." + name + "-", dir=TOOLS / "bin")
                try:
                    with os.fdopen(descriptor, "wb") as output, archive.extractfile(member) as source:
                        while block := source.read(1024 * 1024):
                            output.write(block)
                    os.chmod(temporary, 0o755)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
    print("Project-local Docker binaries ready:", TOOLS / "bin")


def daemon_start():
    if ready():
        print("Project-local rootless Docker already running:", SOCKET)
        return
    if os.getuid() == 0:
        raise RuntimeError("this daemon must run as a regular user")
    launcher_file = RUN / "launcher.pid"
    if launcher_file.exists() and launcher_alive(int(launcher_file.read_text())):
        raise RuntimeError("rootless Docker launcher is still live; inspect its log before starting a second daemon")
    for directory in (RUN, TOOLS / "data", TOOLS / "config"):
        directory.mkdir(parents=True, exist_ok=True)
    RUN.chmod(0o700)
    env = dict(os.environ)
    env["PATH"] = str(TOOLS / "bin") + os.pathsep + env.get("PATH", "")
    env["XDG_RUNTIME_DIR"] = str(RUN)
    daemon_config = TOOLS / "daemon.json"
    daemon_config.write_text("{}\n")
    command = [str(TOOLS / "bin/rootlesskit"), "--state-dir=" + str(RUN / "rootlesskit"),
               "--net=host", "--copy-up=/etc", "--copy-up=/run", "--propagation=rslave",
               "dockerd", "--rootless", "--iptables=false", "--ip6tables=false", "--bridge=none",
               "--userland-proxy=false", "--storage-driver=vfs", "--config-file=" + str(daemon_config),
               "--data-root=" + str(TOOLS / "data"), "--exec-root=" + str(RUN / "exec"),
               "--pidfile=" + str(RUN / "dockerd.pid"), "--host=unix://" + str(SOCKET)]
    with (RUN / "daemon.log").open("a") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
    (RUN / "launcher.pid").write_text(str(process.pid) + "\n")
    for _ in range(100):
        if ready():
            print("Project-local rootless Docker ready:", SOCKET)
            return
        if process.poll() is not None:
            raise RuntimeError("rootless Docker exited; see " + str(RUN / "daemon.log"))
        time.sleep(0.2)
    raise RuntimeError("rootless Docker startup timed out; inspect " + str(RUN / "daemon.log"))


def daemon_stop():
    if ready():
        containers = docker("ps", "-q", capture=True).stdout.split()
        if containers:
            raise RuntimeError("remove running tenant containers before stopping this daemon")
    pidfile = RUN / "launcher.pid"
    if not pidfile.exists():
        raise RuntimeError("daemon was not started by this script; do not stop an unowned process")
    pid = int(pidfile.read_text())
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        pidfile.unlink()
        print("Daemon launcher is already stopped")
        return
    if str(TOOLS / "bin/rootlesskit").encode() not in arguments:
        raise RuntimeError("launcher PID no longer identifies our rootlesskit process")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(100):
        if not launcher_alive(pid):
            pidfile.unlink(missing_ok=True)
            print("Project-local rootless Docker stopped")
            return
        time.sleep(0.1)
    raise RuntimeError("daemon did not stop within 10 seconds")


def name(tenant):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", tenant):
        raise RuntimeError("tenant name must match [a-z][a-z0-9_-]{0,39}")
    return "gpu-tenant-" + tenant


def owned(tenant):
    container = name(tenant)
    details = inspect_container(container)
    if details is None:
        raise RuntimeError("tenant container is absent; run join first")
    require_owned(details)
    return container


def inspect_container(container):
    # Docker holds a container's state lock during parts of create/exit cleanup;
    # VFS and rootless teardown can exceed a short application heartbeat window.
    result = docker("inspect", container, capture=True, check=False)
    if result.returncode:
        message = result.stderr.casefold()
        if "no such object:" in message or "no such container:" in message:
            return None
        raise RuntimeError("cannot inspect tenant container: " + result.stderr.strip())
    return json.loads(result.stdout)[0]


def require_owned(details):
    labels = details["Config"].get("Labels") or {}
    if labels.get("gpu.shared.project") != str(ROOT):
        raise RuntimeError("refusing to operate on a container owned by another project")


@contextmanager
def tenant_lock(tenant):
    # Serialize concurrent join/leave for the same tenant across CLI processes.
    # Docker operations on different tenants remain independent.
    lock_directory = RUN / "tenant-locks"
    lock_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_directory.chmod(0o700)
    descriptor = os.open(lock_directory / (name(tenant) + ".lock"),
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def configuration(args):
    container = name(args.tenant)
    join_timeout = getattr(args, "join_timeout", 10.0)
    if not math.isfinite(join_timeout) or join_timeout <= 0:
        raise RuntimeError("join timeout must be finite and positive")
    info = json.loads(docker("info", "--format", "{{json .}}", capture=True).stdout)
    required = ("MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota")
    if info.get("CgroupVersion") != "2" or not all(info.get(feature) for feature in required):
        raise RuntimeError("tenant limits require working cgroup v2 memory/swap/PID/CPU controllers")
    socket_directory = args.socket_dir.resolve(strict=True)
    token = args.token_file.resolve(strict=True)
    if not socket_directory.is_dir() or not token.is_file():
        raise RuntimeError("socket directory and token file must exist")
    if token.is_relative_to(socket_directory):
        raise RuntimeError("credentials must be outside the shared socket directory")
    entries = list(socket_directory.iterdir())
    if not entries or any(not entry.is_socket() for entry in entries):
        raise RuntimeError("dedicated broker mount directory must contain sockets only; keep tokens and configs elsewhere")
    if not (socket_directory / "gpu.sock").is_socket():
        raise RuntimeError("broker socket must be named gpu.sock inside the dedicated socket directory")
    # Read-only bind mounts do not grant non-root read permission. Credentials
    # are individually readable, while their host parent remains private.
    if token.stat().st_mode & 0o444 != 0o444:
        raise RuntimeError("token file must be mode 0444/0644 so the mapped non-root UID can read it; keep its parent mode 0700")
    return container, socket_directory, token, join_timeout


def join(args):
    container, socket_directory, token, join_timeout = configuration(args)
    with tenant_lock(args.tenant):
        return join_locked(container, socket_directory, token, join_timeout)


def join_locked(container, socket_directory, token, join_timeout):
    existing = inspect_container(container)
    if existing is not None:
        require_owned(existing)
        mounts = {mount["Destination"]: mount for mount in existing.get("Mounts", [])
                  if mount["Type"] == "bind"}
        expected = {"/run/gpu": str(socket_directory), "/run/secrets/tenant.token": str(token)}
        if {destination: mount["Source"] for destination, mount in mounts.items()} != expected or any(
                mount["RW"] for mount in mounts.values()):
            raise RuntimeError("existing tenant mounts differ; leave before joining with new paths")
        if existing["State"]["Running"]:
            # Repeated join does not create a second hold session or interrupt
            # existing work. Verify the current credential and broker first.
            probe = docker("exec", container, "python", "/app/gpuctl.py",
                           "--socket", "/run/gpu/gpu.sock", "--token-file", "/run/secrets/tenant.token",
                           "--timeout", str(min(join_timeout, 2.0)), "heartbeat", capture=True, check=False,
                           timeout=min(join_timeout, 5.0))
            if probe.returncode:
                raise RuntimeError("existing tenant failed broker health check; leave and join again")
            response = json.loads(probe.stdout)
            if not response.get("ok"):
                raise RuntimeError("existing tenant failed broker health check; leave and join again")
            print(json.dumps({"ok": True, "container_id": existing["Id"], "already_joined": True}))
            return existing["Id"]
        docker("rm", existing["Id"], capture=True)
    command = ["run", "--detach", "--name", container, "--label", LABEL,
               "--network", "none", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--user", "1000:1000",
               "--memory", "128m", "--memory-swap", "128m", "--pids-limit", "64", "--cpus", "0.5",
               "--ipc", "private", "--cgroupns", "private",
               "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=16m,uid=1000,gid=1000",
               "--mount", f"type=bind,src={socket_directory},dst=/run/gpu,readonly",
               "--mount", f"type=bind,src={token},dst=/run/secrets/tenant.token,readonly",
               IMAGE, "--socket", "/run/gpu/gpu.sock", "--token-file", "/run/secrets/tenant.token",
               "--timeout", str(min(join_timeout, 5.0)), "hold"]
    # The detached container may fail authentication after docker run succeeds.
    # Require the hold client's actual HELLO response before reporting a join.
    created = None
    try:
        # VFS must copy image layers before the client can authenticate. Keep
        # Docker creation's normal timeout separate from the HELLO deadline.
        created = docker(*command, capture=True, timeout=max(30.0, join_timeout)).stdout.strip()
        deadline = time.monotonic() + join_timeout
        while time.monotonic() < deadline:
            details = inspect_container(created)
            if details is None:
                raise RuntimeError("tenant container disappeared before completing HELLO")
            logs = docker("logs", created, capture=True, check=False, timeout=2)
            lines = logs.stdout.splitlines()
            if lines:
                response = json.loads(lines[0])
                if not response.get("ok"):
                    raise RuntimeError("broker rejected tenant join: " + str(response.get("error", "unknown")))
                if details["State"]["Running"]:
                    print(json.dumps({"ok": True, "container_id": created, "already_joined": False}))
                    return created
            if not details["State"]["Running"]:
                raise RuntimeError("tenant client exited before completing HELLO")
            time.sleep(.1)
        raise RuntimeError("tenant HELLO timed out")
    except BaseException:
        # run may create a container before its CLI times out. Re-inspect only
        # our locked name and label; never remove another project's container.
        try:
            details = inspect_container(created or container)
            if details is not None:
                require_owned(details)
                docker("rm", "--force", details["Id"], capture=True)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as cleanup_error:
            print("container: join rollback failed: " + str(cleanup_error), file=sys.stderr)
        raise


def leave(tenant):
    with tenant_lock(tenant):
        details = inspect_container(name(tenant))
        if details is None:
            print(json.dumps({"ok": True, "already_left": True}))
            return
        require_owned(details)
        if details["State"]["Running"]:
            docker("stop", "--timeout", "5", details["Id"], capture=True)
        docker("rm", details["Id"], capture=True)
        print(json.dumps({"ok": True, "already_left": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("bootstrap", "daemon-start", "daemon-stop", "build", "list", "info"):
        sub.add_parser(command)
    entry = sub.add_parser("join")
    entry.add_argument("tenant")
    entry.add_argument("--socket-dir", type=Path, required=True)
    entry.add_argument("--token-file", type=Path, required=True)
    entry.add_argument("--join-timeout", type=float, default=10.0)
    leave_parser = sub.add_parser("leave")
    leave_parser.add_argument("tenant")
    execute = sub.add_parser("exec")
    execute.add_argument("tenant")
    execute.add_argument("arguments", nargs=argparse.REMAINDER)
    client = sub.add_parser("client")
    client.add_argument("tenant")
    client.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    def terminate(*_):
        # SIGTERM must unwind a join transaction just like Ctrl-C. The outer
        # CLI handler reports interruption after rollback has completed.
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    if args.command == "bootstrap":
        bootstrap()
    elif args.command == "daemon-start":
        daemon_start()
    elif args.command == "daemon-stop":
        daemon_stop()
    elif args.command == "build":
        # Classic builder avoids installing Buildx in the user's global config.
        env = dict(os.environ, DOCKER_BUILDKIT="0")
        executable = TOOLS / "bin/docker"
        # Send only the Dockerfile and client. Never include runtime credentials,
        # daemon sockets, CUDA binaries or unrelated workspace files in context.
        context = io.BytesIO()
        with tarfile.open(fileobj=context, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            archive.add(ROOT / "containers/Dockerfile", arcname="Dockerfile")
            archive.add(ROOT / "python/gpuctl.py", arcname="gpuctl.py")
        subprocess.run([str(executable), "--config", str(TOOLS / "config"), "--host", "unix://" + str(SOCKET),
                        "build", "--network", "none", "-t", IMAGE, "-"],
                       check=True, env=env, input=context.getvalue())
    elif args.command == "join":
        join(args)
    elif args.command == "leave":
        leave(args.tenant)
    elif args.command == "exec":
        if not args.arguments:
            parser.error("exec requires a command")
        interactive = ["-i"] + (["-t"] if sys.stdin.isatty() and sys.stdout.isatty() else [])
        docker("exec", *interactive, owned(args.tenant), *args.arguments, timeout=None)
    elif args.command == "client":
        if not args.arguments:
            parser.error("client requires a gpuctl command")
        interactive = ["-i"] + (["-t"] if sys.stdin.isatty() and sys.stdout.isatty() else [])
        docker("exec", *interactive, owned(args.tenant), "python", "/app/gpuctl.py",
               "--socket", "/run/gpu/gpu.sock", "--token-file", "/run/secrets/tenant.token", *args.arguments, timeout=None)
    elif args.command == "list":
        docker("ps", "--filter", "label=" + LABEL)
    elif args.command == "info":
        docker("info")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("container:", error, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
