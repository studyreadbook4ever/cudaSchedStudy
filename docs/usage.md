<!-- SPDX-License-Identifier: Unlicense -->
# cudaSchedStudy 사용법

한 NVIDIA GPU에 여러 사용자가 참여·작업 제출·취소·퇴장하는 과정을 직접 구현한 CUDA/C++ 실습입니다. Linux host의 broker가 CPU 스케줄링, 고정 크기 GPU 메모리 풀, continuous batching을 관리합니다. 사용자는 Python client 또는 격리된 Docker 컨테이너에서 접속합니다.

직접 작성한 코드는 [Unlicense](../LICENSE)입니다. 외부 CUDA·Docker·Python 소스나 바이너리는 포함하지 않습니다. 외부 도구의 라이선스와 출처는 [THIRD_PARTY.md](../THIRD_PARTY.md)에 있습니다.

이 프로젝트는 LLM serving의 스케줄러와 메모리 수명 관리를 학습하기 위한 **수치 연산 workload**를 실행합니다. 모델 가중치, tokenizer, attention/KV cache는 아직 구현하지 않았습니다. 사용자는 제공된 연산의 크기·반복 횟수·seed를 지정하며, GPU 포인터나 임의 CUDA 커널을 전달하지 않습니다.

## 구현된 동작

- CPU의 weighted deficit 스케줄러와 작업별 round-robin, 작은 GPU 실행 quantum
- 실행 중인 작업 사이에 신규 작업을 넣고 완료·취소된 작업을 제거하는 batching
- lane별 CUDA Green Context, nonblocking stream, 작은 pinned metadata, GPU 내부 초기화와 event polling
- 사용자별 token 인증, 슬롯 quota, 결과 접근 제어, disconnect 시 해당 연결의 작업 정리
- idle 접속은 lane을 점유하지 않으며 작업은 FIFO admission; 완료 이후 같은 stream의 비동기 wipe로 슬롯 재사용
- 사용자별 최근 terminal job 128개 보관, revoke/forget으로 등록·이름 회수
- Docker 파일·프로세스·네트워크 격리와 실제 namespace/cgroup 감사 도구

Green Context는 broker 내부 lane의 SM 집합을 나눕니다. VRAM·L2·메모리 대역폭·GPU fault를 하드웨어적으로 격리하지 않습니다. 다른 host GPU 프로세스를 제한하거나 독점 SM 예약을 만드는 기능도 아닙니다. 자세한 경계는 [설계 문서](architecture.md)에 있습니다.

## 필요한 환경

Linux x86-64, NVIDIA GPU/driver, CUDA Toolkit **13.1 이상**, CMake 3.24 이상, CUDA가 지원하는 C++ compiler, Python 3.10 이상이 필요합니다. Python package 설치는 필요 없습니다. 기본 CUDA target은 compute capability 7.5이며, 다른 GPU에서는 `-DCMAKE_CUDA_ARCHITECTURES=86`처럼 변경합니다.

실제 검증 장비는 GTX 1660 Super 6GB, 22 SM, CUDA 13.3, driver 610.43.03입니다. 기본 4 lane × 4 SM은 broker가 16 SM을 쓰는 설정입니다. 각 lane은 16개 슬롯, 슬롯당 최대 262144 float를 갖고, GPU 데이터 풀은 약 64 MiB입니다. 6GB 전체를 미리 점유하지 않습니다. SM 분할 지원과 granularity는 GPU에 따라 다르며, `green` 모드 초기화 실패 시 종료합니다.

다른 GPU에서는 CUDA architecture와 broker의 `--lanes`·`--sm-per-lane`을 해당 장치에 맞게 설정합니다. 아래 `scripts/verify`의 전체 검증과 고정 benchmark는 이 22 SM 장비의 회귀 검증용입니다. 특히 `backend_stress`는 22 SM을 명시적으로 요구하고 smoke/demo/benchmark도 4 lane × 4 SM을 사용하므로, 다른 GPU의 검증에는 해당 도구의 장치 가정도 조정해야 합니다. CPU client/container/source-export 회귀 테스트는 GPU 없이 실행할 수 있습니다.

Docker는 선택 사항입니다. 제공된 rootless bootstrap은 x86-64 Linux 전용이며, user namespaces, `newuidmap/newgidmap`, subordinate UID/GID 설정이 필요합니다. helper는 cgroup v2 지원이 없으면 참여를 거절하며, 실제 제한값은 감사 도구로 확인합니다.

## 빌드와 첫 실행

```sh
bash scripts/build.sh
build/gpu_probe
build/backend_smoke
build/gpu-host --trace artifacts/host.jsonl
```

`nvcc`를 찾지 못하면 `CUDACXX=/path/to/nvcc bash scripts/build.sh`로 지정합니다. broker는 기본적으로 `runtime/socket/gpu.sock`과 `runtime/private/admin.token`을 만듭니다. 다른 터미널에서 사용자를 등록하고 작업을 실행합니다.

```sh
python3 python/gpuctl.py admin provision alice --quota-slots 4 --weight 1 --token-out runtime/private/alice.token
python3 python/gpuctl.py --token-file runtime/private/alice.token run --elements 65536 --steps 32 --seed 7 --rounds 4
python3 python/gpuctl.py --token-file runtime/private/alice.token shell
python3 python/gpuctl.py admin stats
python3 python/gpuctl.py admin shutdown
```

`shell`은 연결을 유지하면서 `submit ELEMENTS STEPS SEED ROUNDS`, `get ID`, `cancel ID`, `stats`, `leave`를 실행합니다. 연결을 닫으면 그 연결에서 제출한 미완료 작업은 취소됩니다. 같은 사용자의 다른 연결은 계속 사용할 수 있습니다. 등록 정보는 broker 메모리에 있으므로 재시작 후에는 사용자를 다시 등록합니다.

lane이 모두 사용 중이면 작업이 연결을 유지한 채 대기합니다. 마지막 live job이 끝나면 접속 중인 사용자도 lane을 반환합니다. 사용자를 영구 삭제하려면 `admin revoke alice` 후 `admin inspect alice`로 live_slots=0/lane=-1을 확인하고 `admin forget alice`를 사용합니다. 같은 이름을 다시 등록하면 새 token을 받습니다.

`--token-out`은 기존 파일의 덮어쓰기를 거절합니다. broker 재시작이나 FORGET은 디스크의 token 파일을 삭제하지 않으므로 재등록할 때는 새 파일 경로를 사용하거나, 무효화된 이전 token 파일을 먼저 삭제합니다.

장시간 작업과 신규 작업의 혼합, weight 1:3 비교, quota 거절, 취소·퇴장·재참여를 자동 시연하려면 다음을 실행합니다. 실제 GPU를 사용하며 자체 broker를 띄워 정리합니다.

```sh
python3 tools/demo.py --output artifacts/demo.json
```

Docker 사용법과 두 컨테이너에서 실제 CUDA 결과까지 검증하는 `scripts/container-test`는 [containers 문서](containers.md)에 있습니다.

## 검증과 디버깅

```sh
scripts/verify
scripts/container bootstrap
scripts/container daemon-start
scripts/container build
scripts/container-test
scripts/container daemon-stop
```

`scripts/verify`는 source/client/container lifecycle 회귀, 실제 GPU probe, backend smoke/stress, Green/streams 두 모드의 통합 테스트, 데모, Compute Sanitizer 네 가지 검사를 실행합니다. sanitizer가 없는 환경에서는 명시적으로 `--skip-sanitizers`를 사용합니다. `streams`는 SM 분할 없이 일반 stream을 쓰는 비교 모드입니다.

검증 결과와 CUDA-GDB/host sanitizer 재현 명령은 [verification 문서](verification.md), wire protocol과 library 사용법은 [API 문서](api.md)를 참고합니다. runtime, build, token, 로그는 Git에서 제외됩니다.

## GitHub에 소스 공개하기

```sh
scripts/source-audit
scripts/export-source
```

exporter는 검사를 통과한 텍스트 소스만 `dist/gpu-host-lab-source.tar.gz`에 담습니다. LICENSE, 기여 규칙, 의존성 출처와 파일별 SHA256 manifest, 직접 작성한 SVG 구조도가 포함되며, 실행 파일·Docker image·token·개인 실행 로그는 빠집니다. **공개할 때는 이 소스 묶음 또는 `.gitignore`를 적용한 소스 파일을 사용합니다.** GitHub에는 소스만 게시하며 GitHub Actions workflow는 포함하지 않습니다. 루트 README.md에는 작성자의 이론 메모와 아키텍처 설명을 두고, 상세 실행 명령은 이 문서에 보관합니다.

Unlicense는 직접 작성한 코드에 적용됩니다. 향후 외부 코드를 추가하거나 CUDA와 연결된 실행 파일/container image를 배포할 때는 해당 의존성의 재배포 조건을 별도로 확인해야 합니다.

## 코드 읽는 순서

1. [서버](../src/server.cpp): token → 사용자/lane → 작업 → CPU 스케줄러 → 완료 event → 회수
2. [backend 계약](../include/gpu_backend.hpp), [CUDA 구현](../src/gpu_backend.cu): GPU/host 풀과 stream 수명
3. [Python client](../python/gpuctl.py): 연결별 작업 수명과 CLI
4. [통합 테스트](../tests/test_integration.py): 수치 reference, quota, 사용자 경계, batching, 취소·재사용
5. [컨테이너 도구](../containers/manage.py), [격리 감사](../containers/audit.py): GPU device 없는 사용자 환경

새 연산을 추가할 때는 broker가 소유하는 검증된 backend operation으로 등록하고, 작업의 중단 가능한 quantum과 CPU reference부터 정합니다. 실제 LLM을 붙이는 다음 단계에서는 이 메모리 풀을 KV-cache block allocator로 확장하고 prefill/decode를 별도 스케줄링 대상으로 설계할 수 있습니다.
