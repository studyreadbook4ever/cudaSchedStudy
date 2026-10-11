<!-- SPDX-License-Identifier: Unlicense -->
# Contributing

Please keep changes independently authored and under the project's Unlicense.
By submitting a contribution, you dedicate any copyright interest you hold in
that contribution to the public domain under LICENSE. Confirm that you have the
right to make that dedication, including any applicable employer approval.

Do not copy third-party source, CUDA samples, generated SDK headers, or licensed
assets into the repository. Propose dependencies as separately installed tools
and record their provenance in THIRD_PARTY.md. If a contribution requires
third-party material, discuss its exact license and notices before inclusion.

Run the source audit and applicable real-GPU tests. Include the GPU, CUDA version,
backend mode and command used when reporting a GPU-dependent result. A streams
result must not be reported as SM isolation.
