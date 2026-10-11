<!-- SPDX-License-Identifier: Unlicense -->
# 최적화와 사용자 입퇴장

2026-10-11, GTX 1660 Super 6GB / 22 SM / CUDA 13.3 / driver 610.43.03에서 변경 전 실행파일을 보존하고 실제 GPU로 비교했습니다. 아래 수치는 이 합성 affine workload의 결과이며 LLM token 처리량을 뜻하지 않습니다.

## 실행 경로에서 제거한 비용

1. 초기 float 배열을 CPU에서 생성하고 H2D하던 경로를 첫 affine kernel의 초기화로 합쳤습니다. 명시적 `__fmul_rn`/`__fadd_rn`으로 기존 CPU float 초기값의 반올림을 보존합니다. 고정 64 MiB GPU 데이터 풀은 유지하지만 CPU pinned 초기화 공간은 제거했습니다.
2. server의 중간 quantum은 summary kernel/D2H 없이 완료 event만 확인합니다. 최종 quantum은 반드시 수치 summary를 반환합니다. backend의 `report_summary` 기본값은 true이므로 직접 사용하는 기존 호출은 중간 수치를 계속 받을 수 있습니다.
3. 크기가 비슷한 batch는 2D grid를 유지하고, 사각 grid block 수가 유효 block 수의 1.25배를 넘을 때 packed 1D grid를 사용합니다. descriptor 검색은 block당 한 번 수행합니다.
4. 완료된 slot의 전체 wipe는 같은 lane stream에 비동기로 enqueue합니다. 다음 재사용 연산은 stream FIFO 때문에 wipe를 추월하지 못합니다. 첫 실행 전 취소한 예약에는 추가 wipe가 필요 없습니다. CUDA enqueue 실패 시 lane을 poison하고 재사용을 거절합니다.
5. 서버의 tenant별 job/live-job index와 완료 순서 deque가 전역 job scan/history sort를 대체합니다. 숫자 JSON 직렬화는 `std::to_chars`를 사용하고, idle reactor는 매 1 ms 깨어나는 대신 socket과 실제 GPU/tick 상태에 맞춰 기다립니다. tick은 credit refill만 제한하고, 잔여 credit이 있으면 batch 완료 후 다음 tick을 기다리지 않고 dispatch합니다.

## GPU A/B 측정

각 case는 10회 warmup 후 1,000회를 측정했습니다. baseline/optimized를 번갈아 3회 반복했고 아래 완료 시간은 세 run의 p50 중앙값입니다. 4개 Green Context lane의 wave 완료를 측정하며 network/server/client 비용은 제외합니다. GPU clock을 고정하지 않았고 display GPU의 잡음이 있습니다.

| 항목 | 변경 전 | 변경 후 |
| --- | ---: | ---: |
| 신규 큰 job, 4 lane 완료 | 1,584.2 μs | 81.1 μs |
| 균일 크기 resident batch | 315.2 μs | 288.9 μs |
| 혼합 크기 resident batch | 303.8 μs | 128.2 μs |
| 중간 summary 없는 resident batch | 312.4 μs | 264.7 μs |
| 신규 job submit CPU 평균 | 350.4 μs | 7.95 μs |
| pinned host memory | 67,115,008 bytes | 7,168 bytes |

신규 job case는 lane마다 262,144개 원소를 초기화합니다. uniform case는 lane당 16개 × 65,536개 원소이고 mixed case는 15개 × 1개 원소 + 1개 × 262,144개 원소입니다. summary 생략 case는 이전 구현의 기본 summary 반환과 새 구현의 명시적 생략을 비교합니다. workload 형상과 실행 횟수는 `tools/backend_bench.cu`에 고정되어 있습니다.

release CPU 반환 평균은 76.7→2.65 μs였습니다. 이는 동기 대기 제거의 결과이며 GPU zeroing 자체가 그 비율로 빨라졌다는 주장은 하지 않습니다. wipe는 여전히 실행하고 최종 teardown에서 모든 stream을 기다립니다.

원시 결과와 소스 hash는 로컬 `artifacts/backend_optimization_summary.json`과 `artifacts/backend_bench_*_repeat*.jsonl`에 있습니다. baseline은 비교용으로 보존한 최적화 이전 실행파일이며 소스 저장소에는 binary를 포함하지 않습니다. 현재 구현의 측정은 다음으로 재현합니다.

```sh
bash scripts/build.sh
build/backend_bench --repetitions 1000
python3 tools/bench_lifecycle.py --output artifacts/optimization/lifecycle.json
```

측정 중 다른 GPU 테스트나 workload를 병렬로 실행하지 않습니다.

## 접속과 GPU admission 분리

최종 broker/client 경로에서도 변경 전 실행파일과 같은 workload로 비교했습니다. history case는 24 tenant의 terminal job 3,072개를 유지한 뒤 200회 HELLO→LEAVE를 측정합니다. cancel case는 실제 GPU 진행을 확인한 16-job 취소를 10회 반복합니다. trace는 두 측정 모두 껐고 다른 GPU 검증과 순차 실행했습니다.

| 실제 broker 경로 | 변경 전 p50 | 변경 후 p50 |
| --- | ---: | ---: |
| history가 있는 HELLO→LEAVE | 0.1349 ms | 0.0625 ms |
| 16-job 취소 storm의 terminal 도달 | 3.0023 ms | 1.3424 ms |
| 작은 수치 job 제출→완료→퇴장 | 4.8517 ms | 4.6911 ms |
| history 없는 HELLO→LEAVE | 0.0590 ms | 0.0593 ms |

history 입퇴장의 p95는 0.1834→0.1080 ms, 취소의 p95는 3.5963→1.4559 ms였습니다. 이 broker 비교는 각 case를 한 번 실행한 결과이며 client 소스 버전도 각 보고서에 기록합니다. 작은 job의 p99는 5.943→7.653 ms, history 입퇴장의 p99는 0.228→0.233 ms로 tail 개선이 일관되지는 않았습니다. 빈 입퇴장과 작은 job 전체 지연은 큰 개선으로 해석하지 않습니다. GPU microbenchmark의 큰 가속이 socket/tick/event polling을 포함한 모든 serving workload에 같은 비율로 적용되는 것은 아닙니다. `/proc` CPU counter는 10 ms 해상도라 짧은 case의 CPU 0 표시는 비용이 없다는 의미가 아닙니다.

idle 접속은 이전 12명 중 4명 수락·4 lane 점유에서 12명 모두 수락·0 lane 점유로 바뀌었습니다. 새 등록/폐기 benchmark는 128회 뒤 capacity에 도달하던 상태에서 200회 완료·잔여 등록 0으로 바뀌었습니다. 원시 결과는 `artifacts/optimization/baseline-lifecycle.json`, `artifacts/optimization/optimized-lifecycle.json`과 최종 검증 summary에 보관합니다.

인증한 idle 사용자는 lane을 점유하지 않습니다. 첫 job 제출 시 tenant의 논리 quota를 예약하고, 빈 lane이 없다면 FIFO tenant queue에서 기다립니다. 기존 연결로 상태를 조회할 수 있어 capacity 때문에 접속을 반복할 필요가 없습니다. 대기 중 CANCEL/LEAVE/EOF/revoke도 즉시 quota를 반환할 수 있습니다.

같은 tenant의 여러 session은 quota와 lane을 공유합니다. session 종료는 그 session이 제출한 미완료 job만 취소합니다. 마지막 live job이 종료하면 hold session이 남아 있어도 lane을 반환합니다. resident job의 상태를 GPU에서 이동하거나 강제 선점하지 않으므로 긴 job이 lane을 유지하면 대기 tenant의 admission 시간에는 상한이 없습니다.

`reserved_bytes`는 대기 job까지 포함한 논리 예약이며 `allocated_bytes`는 실제 resident slot입니다. 전체 고정 GPU arena가 해제되는 것은 broker 종료 시점입니다. weight는 resident tenant의 연산량을 제어하며 admission queue 순서는 FIFO입니다.

등록을 폐기할 때는 REVOKE로 token/session을 무효화하고 GPU 작업을 drain한 뒤 FORGET을 사용합니다. FORGET은 enabled/session/live-job/lane 조건을 확인해 사용 중인 tenant 포인터를 지우지 않습니다. 삭제 후 같은 이름을 재발급해도 이전 token은 사용할 수 없습니다.

## Client와 Docker lifecycle

timeout·EOF·잘못된 JSON·전송 중 interrupt 이후 client 연결은 닫습니다. 늦게 도착한 이전 응답을 다음 요청에 잘못 연결하지 않습니다. `wait()`의 deadline은 개별 GET의 socket timeout에도 적용합니다.

hold는 작은 인증 HEARTBEAT로 broker 종료/revoke를 감지하고 종료합니다. duplicate join은 기존 container를 그대로 사용하되 heartbeat로 확인하며, duplicate leave는 성공합니다. tenant별 파일 lock이 동시 join/leave를 직렬화합니다. join은 실제 HELLO가 성공할 때까지 확인하고 실패/timeout/interrupt 때 생성한 container를 정리합니다. 다른 프로젝트의 container는 label 검사로 보호합니다.

container audit의 쓰기 가능 검사도 token을 truncate하지 않습니다. 실제 Docker 통합 검사는 lane보다 많은 idle 사용자, 반복 입퇴장, revoke/forget/재발급, 이전 token 거절과 broker 종료 뒤 hold 종료를 검증합니다. 상세 제한은 [containers.md](containers.md), protocol은 [api.md](api.md), 재현과 검증 기록은 [verification.md](verification.md)에 있습니다.
