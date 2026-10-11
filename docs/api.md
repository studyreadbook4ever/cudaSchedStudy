<!-- SPDX-License-Identifier: Unlicense -->
# Broker API

Unix stream socket에 ASCII 공백으로 나눈 한 줄을 보내면 JSON 한 줄이 돌아옵니다. 클라이언트는 응답 순서가 요청 순서와 같다고 가정할 수 있습니다. 요청 최대 4096 byte, 필드 최대 8개, input buffer 최대 64 KiB, output buffer 최대 1 MiB입니다. 인증하지 않은 연결은 5초 뒤 닫습니다. 성공적으로 admin 인증한 연결과 tenant 연결은 유지할 수 있습니다. 입력만 닫는 half-close는 완전한 요청 줄의 응답을 처리하고 연결을 닫습니다. 닫는 연결이 응답을 읽지 않으면 최대 2초 뒤 남은 output을 버리고 회수합니다.

SHUTDOWN/SIGTERM은 GPU 작업과 admission 대기를 취소·drain한 뒤 기존 응답을 nonblocking 전송합니다. 부분 전송/EAGAIN이면 POLLOUT를 기다리되 응답을 읽지 않는 peer 때문에 종료가 무기한 지연되지 않도록 closing deadline을 적용합니다. GPU drain이 끝난 뒤 추가 입력은 처리하지 않습니다.

| 요청 | 권한 / 결과 |
| --- | --- |
| `PING` | 공개: readiness와 backend metadata |
| `ADMIN TOKEN PROVISION NAME SLOTS WEIGHT` | 관리자: tenant 발급, token을 한 번 반환 |
| `ADMIN TOKEN STATS` | 관리자: 전체 tenant/global counts, job 배열은 생략 |
| `ADMIN TOKEN INSPECT NAME` | 관리자: 한 tenant의 상세 정보와 job history |
| `ADMIN TOKEN REVOKE NAME` | 관리자: token 폐기, 세션/미완료 작업 취소 |
| `ADMIN TOKEN FORGET NAME` | 관리자: revoke 후 완전히 drain된 tenant와 history 삭제 |
| `ADMIN TOKEN SHUTDOWN` | 관리자: 신규 작업 거절, drain 후 종료 |
| `HELLO TOKEN` | tenant: 인증된 연결 생성; idle 연결은 GPU lane을 점유하지 않음 |
| `HEARTBEAT` | tenant: 인증 연결의 생존 확인, tenant/lane만 반환 |
| `SUBMIT ELEMENTS STEPS SEED ROUNDS` | tenant: quota 예약, lane 부족 시 FIFO admission queue에 접수 |
| `GET ID` / `CANCEL ID` | 같은 tenant의 작업만 조회/취소 |
| `LIST` / `STATS` | 현재 tenant 작업/사용량과 global aggregate |
| `LEAVE` | 현재 연결의 미완료 작업 취소 후 연결 종료 |

tenant 이름은 영숫자/`_`/`-` 최대 48자, weight 1..16, quota 1..lane slot 수입니다. Docker helper의 이름 규칙은 `[a-z][a-z0-9_-]{0,39}`로 더 좁습니다. 제출 범위는 elements 1..설정된 max, steps 1..100000, seed 0..2^32-1, rounds 1..256입니다. `steps`는 총 진행 step 수, `quantum`은 한 batch에서 진행할 최대 step 수, `rounds`는 원소마다 step 안에서 반복하는 FMA 수입니다.

오류 응답은 `{"ok":false,"error":"CODE"}`입니다. 주요 code는 `unauthorized`, `not_authenticated`, `invalid_request`, `not_found`, `quota_exceeded`, `capacity`, `already_exists`, `server_draining`, `revoked`, `tenant_busy`입니다. 다른 tenant의 작업과 존재하지 않는 작업은 모두 `not_found`입니다. 등록은 최대 128 tenant입니다. `FORGET`은 enabled tenant나 미완료 작업이 남은 tenant에 `tenant_busy`를 반환합니다. revoke → drain 확인 → forget 후에는 같은 이름과 등록 용량을 재사용할 수 있으며 기존 token은 다시 유효해지지 않습니다.

`HELLO`의 `lane`은 `-1`일 수 있습니다. 첫 작업 제출 시 빈 lane을 배정하고, lane이 없으면 연결을 유지한 채 작업을 `queued`로 접수합니다. 대기 job도 quota를 소비하지만 GPU slot은 admission 뒤에 할당합니다. 대기 tenant는 FIFO 순서로 배정하며, 현재 resident 작업을 강제로 선점하거나 GPU 상태를 다른 lane으로 이동하지 않습니다. 긴 작업이 lane을 계속 쓰면 뒤 tenant도 기다립니다. 마지막 live job이 완료·취소되면 session이 남아 있어도 lane을 반환합니다.

tenant/global `live_slots`와 `reserved_bytes`는 admission 대기 작업을 포함한 논리 quota 예약입니다. `resident_slots`(global)와 `allocated_bytes`는 GPU slot을 실제 배정받은 작업만 셉니다. `waiting_for_lane`은 tenant/job admission 대기 여부, `waiting_tenants`는 global 대기 tenant 수입니다. job의 `lane`은 실제 실행 lane 기록이며 완료 후에도 남습니다. 시작 전 취소된 job은 `lane=-1`일 수 있습니다.

job state는 `queued`, `running`, `done`, `cancelled`입니다. 취소 요청의 응답이 `running`일 수 있으므로 terminal state까지 조회해야 합니다. sample/first/last/checksum은 `done`에만 유효합니다. terminal job은 tenant별 최근 128개만 보관하므로 오래된 ID는 `not_found`가 됩니다.

관리자 `STATS`는 128개 tenant의 전체 history가 한 번에 output 한도를 넘지 않도록 counters만 반환하며 `jobs_omitted:true`를 표시합니다. 상세 작업은 `Admin.command("INSPECT", name)`으로 조회합니다. 일반 사용자의 `LIST`/`STATS`에는 자신의 job history가 포함됩니다.

## Python library

`python/gpuctl.py`는 외부 package 없이 import할 수 있습니다.

```python
from pathlib import Path
import sys
sys.path.insert(0, "python")
from gpuctl import Admin, Session

socket = "runtime/socket/gpu.sock"
with Admin.from_file(socket, Path("runtime/private/admin.token")) as admin:
    token = admin.provision("researcher", quota_slots=4, weight=1)["token"]

with Session(socket, token) as session:
    identifier = session.submit(65536, 32, 7, 4)["job_id"]
    result = session.wait(identifier, timeout=30)["job"]
    print(result["state"], result["samples"])
```

CLI는 token을 command line에 직접 넣는 대신 `--token-file`/`--admin-file`로 읽습니다. `admin provision`은 `--token-out` 파일이 필요하며 stdout에서 token을 가립니다. `run`은 결과를 받을 때까지 연결을 유지합니다. 여러 비동기 작업을 직접 제어할 때는 `Session` 또는 `shell`을 사용합니다. token은 bearer credential이므로 사용자 간 공유하지 않습니다.

## 서버 설정

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--socket` | `runtime/socket/gpu.sock` | Unix socket 경로 |
| `--admin-file` | `runtime/private/admin.token` | owned mode 0600 관리자 credential |
| `--mode` | `green` | `green` 분할 필수 / `streams` 일반 stream 비교 |
| `--lanes` | 4 | 최대 동시 resident tenant 수, 1..8; idle/대기 tenant는 제외 |
| `--sm-per-lane` | 4 | Green Context가 요청하는 SM 수 |
| `--slots` | 16 | lane의 고정 슬롯 수 |
| `--max-elements` | 262144 | 슬롯당 float 최대 개수 |
| `--quantum` | 1 | 한 batch에서 job의 최대 진행 step, 1..16 |
| `--batch-elements` | 524288 | 공간 기준 batch 크기 및 tick credit budget |
| `--tick-ms` | 2 | CPU 스케줄링 credit tick, 1..100 ms |
| `--max-clients` | 128 | 연결 수 제한 |
| `--trace` | 없음 | token을 기록하지 않는 JSONL event 로그 |

socket과 admin credential은 서로 다른 디렉토리에 둬야 합니다. 컨테이너에 bind할 socket 디렉토리는 socket만 포함하고 이름은 `gpu.sock`이어야 합니다. 기존 socket을 자동 삭제하지 않으므로 강제 종료 뒤 남은 socket은 실행 중인 broker가 없는 것을 확인한 뒤 수동 정리합니다.
