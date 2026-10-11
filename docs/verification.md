<!-- SPDX-License-Identifier: Unlicense -->
# 실제 로컬 검증

2026-10-11, Linux x86-64 / GTX 1660 Super 6GB / CC 7.5 / 22 SM / NVIDIA driver 610.43.03 / CUDA 13.3에서 실행했습니다. C++ 서버와 CUDA kernel을 직접 컴파일했고 GPU 검증은 실제 장치에서 수행했습니다. client/container의 실패 경로는 별도 CPU 회귀 테스트에서도 검사합니다. 다른 GPU/driver/OS 조합을 검증했다고 주장하지 않습니다. smoke/tests의 기본값은 이 장비의 4 lane × 4 SM 구성입니다.

`scripts/verify` 전체 실행은 위 장비의 회귀 검증입니다. `backend_stress`에는 22 SM 요구 조건과 그 장치의 잘못된 partition 설정 검사가 있으며, smoke/demo/benchmark도 4 lane × 4 SM으로 고정되어 있습니다. 다른 GPU에서는 build architecture, broker 설정과 이 검증 도구의 하드웨어 가정을 함께 조정해야 합니다. CPU 회귀는 GPU와 독립적으로 실행할 수 있습니다.

## 최적화 후 검증

GPU 내부 초기화, summary 선택 생략, packed grid, 비동기 wipe, FIFO admission과 입퇴장 개선 이후 최종 source/binary를 고정하고 다시 실행했습니다. 성능 비교와 변경 내용은 [optimization.md](optimization.md)에 있습니다.

| 검사 | 실제 결과 |
| --- | --- |
| Green Context 통합 | 40/40 PASS, 31.589초 |
| 일반 streams 통합 | 40/40 PASS, 31.367초 |
| CPU AddressSanitizer + UBSan, 실제 GPU 통합 | 40/40 PASS, 41.729초 |
| 정상 broker 종료 | 위 120개 모두 exit 0 |
| 손상 위험 trace 설정 거절 | 5가지 alias/path × 3 suites, 예상 exit 1과 credential/socket 보존 |
| CPU client/container lifecycle/source export | 12 + 16 + 3 tests PASS |
| backend 무작위 계약 stress | 1,400 batches, 3,531 completions, 2,234 할당/반환 일치 |
| summary 반환/생략 | 수치 1,760개 / 수치 없는 completion 1,771개 검증 |
| Compute Sanitizer memcheck/initcheck/racecheck/synccheck | 모두 오류 0, memcheck 누수 0, racecheck 경고 0 |
| 실제 broker lifecycle benchmark | 6개 시나리오, 수치 확인·자원 반환·exit 0 |
| Green/streams 데모 | 각각 9개 검사 PASS, 실제 실행 lane 재사용 확인 |
| 실제 Docker 확장 통합 | 3 idle tenant / 2 lane, 격리·수치·foreign GET 거절·SM 분리 PASS |
| 실제 Docker 반복 입퇴장 | 8회 join·중복 join·GPU 작업·leave·중복 leave, 자원 반환 PASS |
| 실제 Docker credential/종료 수명 | revoke·FORGET·동일 이름 재발급·옛 token 거절·shutdown 뒤 hold 종료 PASS |
| 최종 Docker 부정 설정 | 16개 거절 검사 PASS; cgroup 미지원 응답만 mock |

신규 통합 검사는 12개 idle 연결의 무점유, 1 lane에서 7 tenant의 FIFO admission, 대기 quota와 CANCEL/EOF/reconnect, queued REVOKE/FORGET와 옛 token 거절, 200회 등록·폐기·재발급, admission 전 취소의 history 한도, 180회 session churn을 포함합니다. 큰 STATS 12개와 SHUTDOWN을 pipeline한 delayed reader가 13개 완전한 JSON 응답을 받고, 읽지 않는 peer도 bounded 시간에 종료되는 것을 확인했습니다.

Docker 확장 통합은 별도 2 lane × 2 SM broker에서 실행했고, 한 사용자가 GPU 작업 중 퇴장해도 다른 사용자의 기존 작업과 새 작업이 진행되는 것을 확인했습니다. 8회 재가입과 credential 재발급까지 총 17 job을 실행하고 shutdown 이후 hold의 자동 종료와 생성한 container/credential/broker 정리를 확인했습니다. 최종 source·broker binary·image 안 client의 SHA256을 대조했습니다. 실제 Docker에서 드러난 leave parser 이름 충돌, 소문자 inspect 오류 처리, image 생성과 인증 deadline 혼용, 종료 상태 조회의 과도하게 짧은 timeout도 수정하고 회귀 테스트를 추가했습니다. 근거는 `artifacts/container_integration.json`과 `artifacts/container_isolation.json`입니다.

소스 고정 이전의 중단된 ASan 실행은 성공으로 계산하지 않습니다. 위 표는 최종 binary와 source hash로 끝까지 완료한 검사만 기록합니다. 최신 로컬 증거는 `artifacts/optimization/`의 통합 로그·결과·benchmark·demo와 `final-source-sha256.txt`, `artifacts/backend_optimization_summary.json`, `artifacts/backend_optimized_*.log`입니다. 아래 표와 기존 `artifacts/final/`은 최적화 전 기준입니다.

## 최적화 전 기준 검증

| 검사 | 실제 결과 |
| --- | --- |
| Green Context 통합 테스트 | 30/30 PASS |
| 일반 CUDA stream 비교 모드 | 30/30 PASS; 전체 SM 공유를 명시적으로 확인 |
| CPU AddressSanitizer + UBSan 서버 | 실제 GPU 통합 테스트 30/30 PASS, 모든 daemon exit 0 |
| backend smoke | 수치 reference, 수명 오류 거절, 40회 슬롯 재사용 PASS |
| backend 무작위 계약 stress | 1,400 batches, 3,482 결과, 2,210 할당/반환 PASS |
| Compute Sanitizer 기본 memcheck | 오류 0, 누수 0 byte |
| 기본 initcheck / racecheck / synccheck | 오류 0 / hazard 0 / 오류 0 |
| 전체 broker GPU memcheck | 수치·동시 작업·실행 중 취소·shutdown 4개 시나리오 PASS |
| CUDA-GDB | 실제 kernel entry, CUDA threads/backtrace 확인, 정상 종료 |
| Green/streams 데모 | 각각 9개 동작 검사 PASS |
| 실제 Docker 두 사용자 | OS 격리, CUDA 수치, foreign GET 거절, SM 분리 PASS |
| Docker 부정 설정 | 16개 거절 검사 PASS; cgroup 미지원 응답만 mock |
| 작업 중 컨테이너 퇴장 | Alice 취소/슬롯 반환, Bob 완료, Charlie 교체 참여 PASS |
| source audit/export | SPDX/Unlicense, 외부 도구·runtime credential·binary 제외 |

Green mode의 관찰 SM mask는 `0xf`, `0x3c0`, `0x3c000`, `0x3c00`으로 서로 겹치지 않습니다. stream mode에서는 각 tenant가 `0x3fffff`를 관찰했습니다. credit 제한 workload에서 weight 1:3의 실제 제출 operation 비율은 약 3.01이었습니다. JSON sample과 CPU `fmaf` reference의 차이는 직렬화 반올림 수준이며, backend의 직접 float 비교는 정확히 일치했습니다.

통합 테스트는 인증 실패와 사용자 간 접근 거절, 같은 credential의 quota 공유, lane 최대 참여자 수, 동시 4사용자의 60개 수치 작업, 신규 작업의 배칭, 실제 GPU 실행 중 36회 취소, disconnect/LEAVE/재참여, 최근 terminal history 128개 제한, 96개 pipelined command, 과대/잘못된 요청, CLI credential 파일/hold 수명, revoke, active-work shutdown을 포함합니다.

강화한 검사는 고정 seed의 상태 전환 351회/82개 작업/76회 연결 종료, 잘못된 요청 1,000개, 96개 명령의 EOF drain, 입력 부분 전송, 응답을 읽지 않는 연결의 약 2초 회수, 등록 사용자 128개 한도, queued CANCEL/LEAVE/EOF 이후 history 한도와 오래된 job의 최근 완료 보존을 추가했습니다. 24개 tenant에 queued 취소 3,072개와 실제 수치 작업을 쌓은 뒤 관리자 summary가 약 8 KiB로 유지되고, 24개 `INSPECT` 조회가 정확한 결과를 반환하는 것도 검증했습니다.

backend stress는 350 epoch 동안 4 lane을 동시에 pending으로 유지합니다. pending 중 새 할당 1,400회, 잘못된 호출 8,944회 거절, 잘못된 설정·과대 풀 생성 실패, 미완료 batch가 있는 backend 소멸과 CPU context 복원을 확인했습니다. memcheck는 allocation padding 64 byte와 leak 검사까지 적용했고, 네 종류 sanitizer가 모두 통과했습니다. 결과 검증은 공개 계약인 sample/last/checksum 범위이며, 모든 벡터 원소를 readback하는 검사는 아닙니다.

검토에서 발견한 관리자 연결 5초 만료, half-close 응답 유실, 출력이 막힌 closing 연결 잔류, mixed-cost job starvation은 구현 수정 후 재현 테스트를 통과했습니다. queued 취소의 history 129개 잔류, 제출 ID 순서 때문에 최신 완료 결과를 지우는 문제, 대규모 관리자 통계의 output 한도 초과도 수정했습니다. CUDA runtime Green stream을 supported driver context에 scoped activation하여 기본 sanitizer와 CUDA-GDB의 context mismatch 진단도 해결했습니다. 최종 검사는 `--report-api-errors explicit`로 진단을 줄이지 않고 실행했습니다. CPU/Compute Sanitizer가 프로세스 종료 시 오류를 보고해도 테스트가 실패하도록 모든 daemon 종료코드 0을 검사합니다.

실행 산출물은 `artifacts/`에 보관하며 공개 소스 묶음에서는 제외합니다. 로컬 증거 위치:

- `artifacts/final/client-tests-summary.json`: 두 모드의 검증과 주요 수치
- `artifacts/verify/`: `scripts/verify`의 최신 전체 실행 로그
- `artifacts/final/host-asan-strict.log`, `artifacts/strict/host-asan/*/daemon.log`: CPU sanitizer 실행
- `artifacts/final/daemon-memcheck/`: 전체 broker의 GPU memcheck
- `artifacts/backend_stress*.txt`: GPU backend 계약 stress와 네 가지 sanitizer
- `artifacts/backend_cuda_gdb.txt`: 실제 GPU breakpoint/threads/backtrace
- `artifacts/container_integration.json`, `container_isolation.json`, `container_negative.json`: 실제 컨테이너 검사와 최신 source/image hash 대조
- `artifacts/server_review_regressions.json`: 독립 검토의 edge-case 재현 결과

과거 오류를 담은 디버깅 로그도 남아 있을 수 있습니다. 최종 성공 여부는 위 경로의 최신 결과와 exit code를 사용합니다. 이 기록은 테스트한 workload와 환경의 근거이며, 임의 커널이나 적대적인 GPU 코드 실행의 안전성 증명이 아닙니다.

## 직접 재현

```sh
scripts/verify
scripts/container bootstrap
scripts/container daemon-start
scripts/container build
scripts/container-test
scripts/container daemon-stop
```

컨테이너 검사는 자체 broker와 컨테이너를 만들고 정리합니다. rootless cache의 외부 도구와 image는 재사용을 위해 남습니다. `scripts/verify`는 이미 설치한 CUDA를 사용하고 시스템 설정을 수정하지 않습니다.

CPU 서버 sanitizer는 별도 build에서 실행할 수 있습니다. CUDA kernel의 검사 도구는 Compute Sanitizer이며, 이 CPU sanitizer 옵션은 C++ 서버에 적용됩니다.

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-asan -j4
ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=1:abort_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 GPU_HOST_BINARY="$PWD/build-asan/gpu-host" \
  GPU_TEST_ARTIFACT_DIR="$PWD/artifacts/host-asan" \
  python3 -m unittest discover -s tests -v
```

이 장비에서는 ASan 기본 shadow-gap 보호가 CUDA 초기화와 충돌하여 `select CUDA device: out of memory`가 발생했습니다. GPU 메모리가 부족한 상황은 아니었고, `protect_shadow_gap=0`을 적용한 동일 build에서 통과했습니다. CUDA와 ASan을 함께 사용할 때 이 옵션을 안내하는 [NVIDIA 문서](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/troubleshooting.html)를 참고했습니다.

GPU kernel 디버깅은 다음처럼 재현합니다. CUDA Toolkit의 bin을 PATH에 추가하거나 절대 경로를 사용합니다.

```sh
cuda-gdb --batch \
  -ex 'set pagination off' -ex 'set cuda break_on_launch application' \
  -ex run -ex 'info cuda kernels' -ex 'info cuda threads' -ex bt \
  -ex 'set cuda break_on_launch none' -ex continue build/backend_smoke
```

공개용 archive는 `scripts/export-source`로 생성합니다. archive를 새 디렉토리에 풀어 같은 장비에서 다시 빌드하고 smoke를 실행하는 것으로 원래 workspace 경로에 의존하지 않음을 확인합니다.
