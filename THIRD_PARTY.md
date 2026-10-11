<!-- SPDX-License-Identifier: Unlicense -->
# External dependencies and provenance

The source files in this repository were written for this project and are
released under the Unlicense in LICENSE. No CUDA SDK headers, NVIDIA sample
implementations, Docker binaries, base-image filesystems, or third-party source
libraries are vendored in the source distribution.

Using an external dependency does not change that dependency's license.

| Dependency | How it is used | Upstream terms |
| --- | --- | --- |
| NVIDIA CUDA Toolkit and driver | Installed separately; headers at compilation and dynamic CUDA runtime/driver at execution | [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) |
| NVIDIA Compute Sanitizer and CUDA-GDB | Optional, separately installed verification tools | Distributed by NVIDIA under their own terms |
| CMake and a C++ compiler | Separately installed build tools | Their upstream licenses |
| Python | Separately installed standard-library-only client/tests; also in an externally pulled image | [Python license](https://docs.python.org/3/license.html) |
| Docker Engine / Moby | Optional rootless container runtime, downloaded into a user cache outside this source tree | [Moby license and notices](https://github.com/moby/moby/blob/master/LICENSE) |
| RootlessKit and rootless runtime helpers | Included in the external Docker rootless tools distribution, kept outside the source tree | Their respective upstream licenses and notices |
| Python/Alpine container image | Pulled by digest from the upstream registry, never bundled in the source distribution | [Docker Python image source](https://github.com/docker-library/python), [Alpine licensing](https://www.alpinelinux.org/about/) |

API names and documented behavior were consulted in NVIDIA documentation. The
broker, CUDA workload, client, tests and container-management scripts were
implemented independently. They do not incorporate a CUDA sample's source code.

The provided release exporter packages source only. Compiled binaries and
container images require a separate dependency/redistribution review if someone
later chooses to distribute them. The project license does not relicense them.
