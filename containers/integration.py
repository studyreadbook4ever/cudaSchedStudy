#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Run two actual Docker tenants against this machine's compiled CUDA broker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

from manage import ROOT, TOOLS, SOCKET, docker, inspect_container, join, leave, name, owned

sys.path.insert(0, str(ROOT / "python"))
from gpuctl import Admin, HostError, Session

WORKER = r'''
import ctypes,json,math,sys
from pathlib import Path
sys.path.insert(0,'/app')
from gpuctl import Session
lib=ctypes.CDLL(None)
lib.fmaf.argtypes=[ctypes.c_float]*3
lib.fmaf.restype=ctypes.c_float
def f32(v): return ctypes.c_float(v).value
def reference(seed,index,iterations):
    x=f32(f32(f32(seed%97)*f32(.001))+f32(f32(index%97)*f32(.001)))
    add=f32(f32(.0001)*f32(seed%97))
    for _ in range(iterations): x=lib.fmaf(x,f32(.9999),add)
    return x
token=Path('/run/secrets/tenant.token').read_text().strip()
seed=int(sys.argv[1])
with Session('/run/gpu/gpu.sock',token) as session:
    print(json.dumps({'stage':'ready','tenant':session.hello['tenant']}),flush=True)
    assert sys.stdin.readline().strip()=='go'
    a=session.submit(65536,128,seed,1)
    b=session.submit(1024,64,seed+1,1)
    print(json.dumps({'stage':'submitted','job_ids':[a['job_id'],b['job_id']]}),flush=True)
    foreign=int(sys.stdin.readline())
    forbidden=session.request('GET',foreign,check=False)
    assert forbidden=={'ok':False,'error':'not_found'},forbidden
    results=[]
    max_error=0.0
    for identifier in [a['job_id'],b['job_id']]:
        job=session.wait(identifier,30)['job']
        assert job['state']=='done',job
        iterations=job['total_steps']*job['rounds']
        expected=[reference(job['seed'],i,iterations) for i in range(len(job['samples']))]
        pairs=list(zip(job['samples'],expected))
        pairs += [(job['first'],reference(job['seed'],0,iterations)),(job['last'],reference(job['seed'],job['elements']-1,iterations))]
        for actual,wanted in pairs:
            max_error=max(max_error,abs(actual-wanted))
            assert math.isclose(actual,wanted,rel_tol=1e-9,abs_tol=1e-9),(actual,wanted)
        assert job['sm_mask']!=0,job
        results.append(job)
    print(json.dumps({'stage':'verified','tenant':session.hello['tenant'],'foreign_get':forbidden,
                      'numeric_max_absolute_error':max_error,'jobs':results}),flush=True)
'''

CHURN_WORKER = r'''
import ctypes,json,math,sys,time
from pathlib import Path
sys.path.insert(0,'/app')
from gpuctl import Session
with Session('/run/gpu/gpu.sock',Path('/run/secrets/tenant.token').read_text().strip()) as session:
    abandon=sys.argv[1]=='abandon'
    submitted=session.submit(65536,100000,91 if abandon else 92,64 if abandon else 1)
    deadline=time.monotonic()+10
    while True:
        job=session.get(submitted['job_id'])['job']
        if job['completed_steps']>0: break
        assert time.monotonic()<deadline,'job never reached the actual GPU'
        time.sleep(.01)
    print(json.dumps({'stage':'active','job':job}),flush=True)
    if abandon:
        sys.stdin.readline()
    else:
        # The coordinator confirms actual departure before releasing this
        # barrier. A finite initial job would depend on Docker stop latency.
        assert sys.stdin.readline().strip()=='departed'
        continued=session.get(submitted['job_id'])['job']
        assert continued['state'] in {'queued','running'},continued
        assert continued['completed_steps']>job['completed_steps'],continued
        finite=session.submit(1024,128,92,1)
        result=session.wait(finite['job_id'],30)
        assert result['job']['state']=='done',result
        lib=ctypes.CDLL(None)
        lib.fmaf.argtypes=[ctypes.c_float]*3
        lib.fmaf.restype=ctypes.c_float
        f32=lambda value: ctypes.c_float(value).value
        for index,actual in enumerate(result['job']['samples']):
            expected=f32(f32(92*f32(.001))+f32(index*f32(.001)))
            for _ in range(128): expected=lib.fmaf(expected,f32(.9999),f32(92*f32(.0001)))
            assert math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9),(actual,expected)
        session.cancel(submitted['job_id'])
        cancelled=session.wait(submitted['job_id'],30)['job']
        assert cancelled['state']=='cancelled',cancelled
        print(json.dumps({'stage':'survived','job':result['job'],'continued_background':continued,
                          'background_cancelled':cancelled}),flush=True)
'''


def worker(tenant, seed, source=WORKER):
    command = [str(TOOLS / "bin/docker"), "--config", str(TOOLS / "config"), "--host", "unix://" + str(SOCKET),
               "exec", "-i", owned(tenant), "python", "-c", source, str(seed)]
    return subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_stage(process, expected):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=40):
            raise TimeoutError("container worker did not reach stage " + expected)
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("container worker exited: " + process.stderr.read())
    response = json.loads(line)
    assert response.get("stage") == expected, response
    return response


def wait_stopped(tenant, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        details = inspect_container(name(tenant))
        if details is None or not details["State"]["Running"]:
            return
        time.sleep(.05)
    raise TimeoutError("hold container did not detect broker shutdown/revocation")


def write_token(path, token):
    path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(token + "\n")
    path.chmod(0o444)


def run_job(tenant, elements=263, steps=7, seed=71):
    answer = docker("exec", owned(tenant), "python", "/app/gpuctl.py", "--socket", "/run/gpu/gpu.sock",
                    "--token-file", "/run/secrets/tenant.token", "run", "--elements", str(elements),
                    "--steps", str(steps), "--seed", str(seed), capture=True)
    responses = [json.loads(line) for line in answer.stdout.splitlines()]
    assert responses[-1]["job"]["state"] == "done", responses
    return responses[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/container_integration.json")
    args = parser.parse_args()
    # Test credentials never reuse or replace a user's existing runtime files.
    fixture = tempfile.TemporaryDirectory(prefix="gpu-container-integration-")
    runtime = Path(fixture.name)
    socketdir = runtime / "socket"
    private = runtime / "private"
    socketdir.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    socketdir.chmod(0o755)
    private.chmod(0o700)
    host_socket = socketdir / "gpu.sock"
    adminfile = private / "admin.token"
    trace = ROOT / "artifacts/container-host.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    children = []
    joined = []
    suffix = uuid.uuid4().hex[:8]
    alice, bob, charlie = ("alice_it_" + suffix, "bob_it_" + suffix, "charlie_it_" + suffix)
    admin = None
    report = {"ok": False}
    source_paths = ("python/gpuctl.py", "containers/manage.py", "containers/audit.py", "containers/integration.py",
                    "src/server.cpp", "src/gpu_backend.cu", "include/gpu_backend.hpp", "containers/Dockerfile")
    source_sha256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}
    binary_sha256 = hashlib.sha256((ROOT / "build/gpu-host").read_bytes()).hexdigest()
    with (trace.parent / "container-host.log").open("w") as log:
        host = subprocess.Popen([str(ROOT / "build/gpu-host"), "--socket", str(host_socket),
                                 "--admin-file", str(adminfile), "--lanes", "2", "--sm-per-lane", "2",
                                 "--slots", "4", "--max-elements", "65536", "--trace", str(trace)], stdout=log, stderr=log)
    try:
        for _ in range(200):
            if host.poll() is not None:
                raise RuntimeError("CUDA broker exited; see artifacts/container-host.log")
            if host_socket.exists() and adminfile.exists():
                try:
                    admin = Admin.from_file(str(host_socket), adminfile)
                    admin.stats()
                    break
                except OSError:
                    pass
            time.sleep(.05)
        if admin is None:
            raise RuntimeError("CUDA broker startup timed out")
        for tenant in (alice, bob, charlie):
            issued = admin.provision(tenant, 4, 1)
            tokenfile = private / (tenant + ".token")
            write_token(tokenfile, issued["token"])
        admin.close()
        admin = None
        # Three idle tenants can join a two-lane GPU: authentication no longer
        # consumes a GPU lane until there is actual work to execute.
        for tenant in (alice, bob, charlie):
            join(SimpleNamespace(tenant=tenant, socket_dir=socketdir, token_file=private / (tenant + ".token")))
            joined.append(tenant)
        image_clients = {}
        for tenant in (alice, bob, charlie):
            actual = docker("exec", owned(tenant), "python", "-c",
                            "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('/app/gpuctl.py').read_bytes()).hexdigest())", capture=True).stdout.strip()
            assert actual == source_sha256["python/gpuctl.py"], "Docker image contains an outdated client; run scripts/container build"
            image_clients[tenant] = actual
        subprocess.run([str(ROOT / "scripts/container-audit"), alice, bob, charlie], check=True)
        children = [worker(alice, 17), worker(bob, 33)]
        ready = [read_stage(process, "ready") for process in children]
        for process in children:
            process.stdin.write("go\n")
            process.stdin.flush()
        submitted = [read_stage(process, "submitted") for process in children]
        for index, process in enumerate(children):
            process.stdin.write(str(submitted[1-index]["job_ids"][0]) + "\n")
            process.stdin.flush()
        results = [read_stage(process, "verified") for process in children]
        for process in children:
            assert process.wait(timeout=10) == 0, process.stderr.read()
        masks = []
        for result in results:
            mask = 0
            for job in result["jobs"]:
                mask |= job["sm_mask"]
            masks.append(mask)
        assert masks[0] & masks[1] == 0, "different Docker tenants executed on overlapping SM sets"
        admin = Admin.from_file(str(host_socket), adminfile)
        before_leave = admin.stats()
        assert before_leave["global"]["backend"]["mode"] == "green-contexts"
        assert before_leave["global"]["multi_job_batches"] > 0, "no actual multi-job batch was observed"
        admin.close()
        admin = None
        departing = worker(alice, "abandon", CHURN_WORKER)
        continuing = worker(bob, "continue", CHURN_WORKER)
        children.extend([departing, continuing])
        departing_job = read_stage(departing, "active")["job"]
        continuing_job = read_stage(continuing, "active")["job"]
        subprocess.run([str(ROOT / "scripts/container"), "leave", alice], check=True)
        joined.remove(alice)
        assert departing.wait(timeout=10) != 0, "departing tenant's active process was not terminated"
        continuing.stdin.write("departed\n")
        continuing.stdin.flush()
        survivor = read_stage(continuing, "survived")
        survived = survivor["job"]
        assert continuing.wait(timeout=10) == 0, continuing.stderr.read()
        with Admin.from_file(str(host_socket), adminfile) as observer:
            after_leave = observer.stats()
            alice_state = observer.inspect(alice)["tenant"]
        assert after_leave["jobs_omitted"] and all(not tenant["jobs"] for tenant in after_leave["tenants"])
        cancelled = next(job for job in alice_state["jobs"] if job["job_id"] == departing_job["job_id"])
        assert cancelled["state"] == "cancelled" and alice_state["live_slots"] == 0, alice_state
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        cancel_order = next(i for i,event in enumerate(events) if event.get("event")=="job_terminal" and event.get("job_id")==departing_job["job_id"])
        done_order = next(i for i,event in enumerate(events) if event.get("event")=="job_terminal" and event.get("job_id")==survived["job_id"])
        assert cancel_order < done_order, "surviving work did not complete after the other tenant departed"
        # An already authenticated idle container can consume a released lane.
        replacement = run_job(charlie)
        churn = []
        for cycle in range(8):
            config = SimpleNamespace(tenant=alice, socket_dir=socketdir, token_file=private / (alice + ".token"))
            first = join(config)
            joined.append(alice)
            second = join(config)
            assert first == second, "repeated join created another container"
            result = run_job(alice, elements=263, steps=7, seed=cycle + 1)
            leave(alice)
            joined.remove(alice)
            leave(alice)
            with Admin.from_file(str(host_socket), adminfile) as observer:
                deadline = time.monotonic() + 5
                while True:
                    state = observer.inspect(alice)["tenant"]
                    if state["active_sessions"] == 0 and state["live_slots"] == 0 and state["lane"] == -1:
                        break
                    assert time.monotonic() < deadline, state
                    time.sleep(.01)
            churn.append({"cycle": cycle, "job_id": result["job"]["job_id"], "container_reused": True,
                          "sessions_after_leave": state["active_sessions"], "live_slots_after_leave": state["live_slots"],
                          "lane_after_leave": state["lane"]})
        old_token = (private / (charlie + ".token")).read_text().strip()
        with Admin.from_file(str(host_socket), adminfile) as observer:
            observer.revoke(charlie)
            wait_stopped(charlie)
            observer.forget(charlie)
            issued = observer.provision(charlie, 4, 1)
        try:
            Session(str(host_socket), old_token).close()
        except HostError as error:
            assert error.response["error"] == "unauthorized", error.response
        else:
            raise AssertionError("forgotten tenant's old credential was accepted")
        write_token(private / (charlie + ".token"), issued["token"])
        join(SimpleNamespace(tenant=charlie, socket_dir=socketdir, token_file=private / (charlie + ".token")))
        recycled = run_job(charlie, seed=83)
        report = {"ok": False, "tenants": ready, "results": results,
                  "sm_masks_disjoint": True, "multi_job_batches": before_leave["global"]["multi_job_batches"],
                  "departure_during_other_tenant_work": {"cancelled": cancelled, "survived": survived,
                                                         "continued_background": survivor["continued_background"],
                                                         "background_cancelled": survivor["background_cancelled"],
                                                         "cancelled_before_other_finished": True},
                  "replacement_tenant_completed": replacement, "backend": before_leave["global"]["backend"],
                  "idle_tenants_exceed_lanes": {"authenticated_tenants": 3, "gpu_lanes": 2},
                  "repeated_join_leave": churn,
                  "credential_lifecycle": {"revoked_hold_exited": True, "old_token_rejected": True,
                                           "forgotten_name_reprovisioned": True, "new_job": recycled}}
        assert source_sha256 == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}, "tested source changed during the run; rerun against final source"
        assert binary_sha256 == hashlib.sha256((ROOT / "build/gpu-host").read_bytes()).hexdigest(), "broker executable changed during the run"
        report["source_verification"] = {"source_sha256": source_sha256, "broker_binary_sha256": binary_sha256,
                                         "image_client_sha256": image_clients, "image_matches_current_source": True,
                                         "source_unchanged_during_run": True}
        with Admin.from_file(str(host_socket), adminfile) as observer:
            observer.shutdown()
        for tenant in joined:
            wait_stopped(tenant)
        report["broker_shutdown_detected_by_holds"] = True
        report["ok"] = True
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        for tenant in joined:
            docker("rm", "--force", owned(tenant), check=False)
        if admin:
            admin.close()
        if host.poll() is None:
            try:
                admin = Admin.from_file(str(host_socket), adminfile)
                admin.shutdown()
            except OSError:
                pass
            finally:
                if admin:
                    admin.close()
        try:
            host.wait(timeout=10)
        except subprocess.TimeoutExpired:
            host.terminate()
            host.wait(timeout=10)
        if report["ok"]:
            assert host.returncode == 0, "CUDA broker did not exit cleanly"
            report["cleanup"] = {"broker_exit_code": host.returncode, "containers_removed": True,
                                 "temporary_credentials_removed": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        fixture.cleanup()
    print(json.dumps({"ok": True, "report": str(args.output), "sm_masks": masks,
                      "multi_job_batches": report["multi_job_batches"]}))


if __name__ == "__main__":
    main()
