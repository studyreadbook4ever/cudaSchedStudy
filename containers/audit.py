#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Inspect actual running tenant namespaces, mounts, capabilities and cgroups."""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys

from manage import ROOT, docker, owned

PROBE = r'''
import errno,glob,json,os,socket
from pathlib import Path
status = {}
for line in Path('/proc/self/status').read_text().splitlines():
    if ':' in line:
        k,v = line.split(':',1)
        status[k] = v.strip()
def writable(path,existing=False):
    try:
        # Test permission without truncating a real credential if a mount is
        # accidentally writable. Temporary root probes are removed immediately.
        fd=os.open(path,os.O_WRONLY|(0 if existing else os.O_CREAT|os.O_EXCL),0o600)
        os.close(fd)
        if not existing: os.unlink(path)
    except OSError as error:
        return {'blocked': True, 'errno': error.errno}
    return {'blocked': False}
internet = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
internet.settimeout(0.5)
try:
    internet.connect(('1.1.1.1',443))
    external_network = True
except OSError:
    external_network = False
finally:
    internet.close()
print(json.dumps({
 'uid':os.getuid(),'gid':os.getgid(),
 'namespaces':{k:os.readlink('/proc/self/ns/'+k) for k in ['pid','mnt','net','ipc']},
 'status':{k:status[k] for k in ['CapEff','NoNewPrivs','Seccomp']},
 'gpu_devices':glob.glob('/dev/nvidia*')+glob.glob('/dev/dri/*'),
 'docker_socket_visible':Path('/var/run/docker.sock').exists(),
 'interfaces':socket.if_nameindex(),'external_network':external_network,
 'secret_names':[p.name for p in Path('/run/secrets').iterdir()],
 'root_write':writable('/app/audit-forbidden-write-'+os.urandom(8).hex()),
 'secret_write':writable('/run/secrets/tenant.token',existing=True),
 'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
 'pids_max':Path('/sys/fs/cgroup/pids.max').read_text().strip(),
 'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tenants", nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/container_isolation.json")
    args = parser.parse_args()
    if len(args.tenants) < 2:
        parser.error("at least two running tenants are required to prove separation")
    reports = {}
    host_ns = {key: os.readlink("/proc/self/ns/" + key) for key in ("pid", "mnt", "net", "ipc")}
    for tenant in args.tenants:
        container = owned(tenant)
        inspect = json.loads(docker("inspect", container, capture=True).stdout)[0]
        actual = json.loads(docker("exec", container, "python", "-c", PROBE, capture=True).stdout)
        hostconfig = inspect["HostConfig"]
        assert inspect["State"]["Running"], tenant + " must be running"
        assert actual["uid"] == actual["gid"] == 1000
        assert actual["status"] == {"CapEff": "0000000000000000", "NoNewPrivs": "1", "Seccomp": "2"}
        assert not actual["gpu_devices"] and not actual["docker_socket_visible"]
        assert actual["interfaces"] == [[1, "lo"]] and not actual["external_network"]
        assert actual["secret_names"] == ["tenant.token"]
        assert actual["root_write"]["blocked"] and actual["secret_write"]["blocked"]
        assert actual["memory_max"] == str(128 * 1024 * 1024)
        assert actual["pids_max"] == "64"
        assert actual["cpu_max"].split()[0] == "50000"
        assert hostconfig["ReadonlyRootfs"] and not hostconfig["Privileged"]
        assert hostconfig["NetworkMode"] == "none" and hostconfig["CapDrop"] == ["ALL"]
        assert not hostconfig["Devices"] and not hostconfig["DeviceRequests"]
        assert {m["Destination"] for m in inspect["Mounts"] if m["Type"] == "bind"} == {"/run/gpu", "/run/secrets/tenant.token"}
        assert all(not m["RW"] for m in inspect["Mounts"] if m["Type"] == "bind")
        for key, namespace in actual["namespaces"].items():
            assert namespace != host_ns[key], f"{tenant} shares host {key} namespace"
        mounts = [{"type": mount["Type"], "destination": mount["Destination"],
                   "readonly": not mount["RW"]} for mount in inspect["Mounts"]]
        reports[tenant] = {"container_id": inspect["Id"], "runtime": actual,
                           "mounts": mounts, "host_pid": inspect["State"]["Pid"]}
    for key in host_ns:
        assert len({report["runtime"]["namespaces"][key] for report in reports.values()}) == len(reports), key + " namespaces overlap"
    marker = "audit-" + secrets.token_hex(8)
    first = args.tenants[0]
    docker("exec", owned(first), "python", "-c", "from pathlib import Path; Path('/tmp/" + marker + "').write_text('tenant-private')", capture=True)
    for tenant in args.tenants[1:]:
        answer = docker("exec", owned(tenant), "python", "-c", "from pathlib import Path; print(Path('/tmp/" + marker + "').exists())", capture=True)
        assert answer.stdout.strip() == "False", "tenant-private tmpfs leaked"
    docker("exec", owned(first), "python", "-c", "from pathlib import Path; Path('/tmp/" + marker + "').unlink()", capture=True)
    result = {"ok": True, "host_namespaces": host_ns, "tenant_filesystem_separation": True,
              "tenants": reports, "scope": "OS isolation; broker authorization is tested separately"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": True, "tenants": args.tenants, "report": str(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        print("container audit:", error, file=sys.stderr)
        sys.exit(1)
