<!-- SPDX-License-Identifier: Unlicense -->
# 한 GPU를 여러 Docker 사용자 환경에서 쓰기

각 사용자는 실제 Docker 컨테이너 안에서 CLI와 자신의 파일을 사용합니다. 컨테이너에는 `/dev/nvidia*`, 호스트 CUDA 라이브러리, Docker socket을 전달하지 않습니다. 사용자가 보낸 작업은 Unix socket을 통해 호스트의 CUDA broker로 들어갑니다. CPU 스케줄러, quota와 GPU buffer 관리는 broker가 수행하고, 컨테이너에는 그 사용자의 결과만 돌아옵니다.

Docker가 파일시스템·프로세스·네트워크를 분리하고, broker가 GPU 작업과 결과 접근을 관리하는 구조입니다. 이 GTX 1660 Super 실습은 컨테이너마다 하드웨어 GPU를 하나씩 만드는 구조가 아닙니다. 사용자에게 임의 CUDA 포인터나 임의 커널 실행 권한을 주지 않고, broker가 제공하는 연산의 크기·반복 횟수 등을 지정하게 합니다.

## 시스템 설치 없이 Docker 준비

이 머신에는 Docker daemon이 없었지만, user namespace와 subordinate UID/GID 설정이 있어서 rootless Docker를 저장소 바깥의 사용자 cache에 내려받아 실제로 실행할 수 있었습니다. root 권한, 패키지 설치, 사용자 systemd 서비스 등록을 사용하지 않습니다. Docker CLI의 설정도 그 cache에 따로 두어서 기존 개인 Docker 설정을 변경하지 않습니다.

```sh
scripts/container bootstrap
scripts/container daemon-start
scripts/container build
```

고정 버전의 공식 Docker static archive를 SHA256으로 확인하며, tenant 이미지는 공식 Python Alpine 이미지의 digest에 고정합니다. image build context에는 Dockerfile과 `python/gpuctl.py` 두 파일만 들어갑니다. binary·image·daemon 로그는 저장소 바깥의 `${XDG_CACHE_HOME:-~/.cache}/gpu-host-lab/docker`에 저장됩니다. `scripts/container info`로 실제 daemon을 확인할 수 있습니다. 이 저장소의 직접 작성한 코드는 Unlicense이며, Docker/RootlessKit/Python/Alpine/CUDA 같은 외부 의존성의 원래 라이선스는 그대로 적용됩니다. 외부 binary와 image를 소스 저장소에 복사하거나 함께 재배포하지 않습니다.

rootless daemon 자체는 이미지를 내려받기 위해 호스트 네트워크를 사용합니다. **사용자 컨테이너는 항상 `--network none`으로 생성됩니다.** 메모리와 PID 제한이 실제로 적용되려면 host의 cgroup v2/systemd 구성이 필요합니다. `join`은 daemon이 memory/swap/PID/CPU 제한 지원을 보고하지 않으면 실행을 거절합니다. 이 머신에서는 audit가 `/sys/fs/cgroup/memory.max`, `pids.max`, `cpu.max`의 값을 직접 확인합니다.

## broker와 사용자 자격증명 준비

공유할 socket 디렉토리에는 Unix socket만 있어야 합니다. admin token과 사용자 token은 별도의 private 디렉토리에 둡니다.

```sh
mkdir -p runtime/socket runtime/secrets
chmod 0755 runtime/socket
chmod 0700 runtime/secrets
build/gpu-host --socket runtime/socket/gpu.sock --admin-file runtime/secrets/admin.token
```

다른 터미널에서 사용자를 등록합니다.

```sh
python3 python/gpuctl.py --socket runtime/socket/gpu.sock --admin-file runtime/secrets/admin.token admin provision alice --quota-slots 4 --weight 1 --token-out runtime/secrets/alice.token
python3 python/gpuctl.py --socket runtime/socket/gpu.sock --admin-file runtime/secrets/admin.token admin provision bob --quota-slots 4 --weight 1 --token-out runtime/secrets/bob.token
chmod 0444 runtime/secrets/alice.token runtime/secrets/bob.token
```

사용자 token 파일은 컨테이너의 nonroot UID가 읽도록 `0444`로 설정합니다. 호스트의 `runtime/secrets` 디렉토리는 `0700`으로 유지합니다. 컨테이너에는 그 사용자의 token 파일 하나만 read-only mount됩니다. admin token과 다른 사용자 token은 mount하지 않습니다.

## 참여, 작업 실행, 나가기

```sh
scripts/container join alice --socket-dir runtime/socket --token-file runtime/secrets/alice.token
scripts/container join bob --socket-dir runtime/socket --token-file runtime/secrets/bob.token
scripts/container list
scripts/container client alice run --elements 262144 --steps 32 --seed 1 --rounds 32
scripts/container client bob run --elements 262144 --steps 32 --seed 2 --rounds 32
scripts/container client alice shell
scripts/container leave alice
```

두 터미널에서 Alice/Bob의 `run`을 동시에 실행하면 같은 GPU broker에 다른 사용자 작업이 들어갑니다. `shell`은 하나의 연결을 유지하면서 여러 작업을 제출·조회·취소하는 대화형 CLI입니다. 일반 사용자 환경을 확인하려면 `scripts/container exec alice sh`를 사용합니다. 해당 컨테이너의 `/tmp`는 그 사용자만 쓰는 제한된 tmpfs입니다.

`join`은 `hold` 연결을 유지하는 컨테이너를 띄웁니다. `hold`는 2초마다 인증된 `HEARTBEAT`를 보내므로 broker가 종료되거나 credential이 폐기되면 자동으로 종료합니다. 이 응답에는 tenant 이름과 현재 lane만 들어가므로 긴 작업 이력을 반복해서 전송하지 않습니다. SIGTERM을 받으면 `LEAVE` 후 연결을 닫고, 아직 `HELLO`를 기다리는 중이라면 즉시 연결을 닫습니다. 인증된 idle 연결은 GPU lane을 차지하지 않습니다. 실제 작업이 제출될 때 lane을 배정하고 모든 작업이 끝나면 연결이 남아 있어도 반환하므로, GPU lane 수보다 많은 사용자 환경을 미리 띄워둘 수 있습니다. lane 부족 시 제출한 작업은 tenant quota를 예약하고 FIFO admission queue에서 대기합니다.

같은 tenant에 `join`을 다시 실행하면 mount 경로·credential·broker 건강 상태를 확인한 뒤 기존 실행 중인 컨테이너를 사용합니다. 추가 `hold` 연결을 만들거나 진행 중인 작업을 취소하지 않습니다. 이전에 종료된 컨테이너는 제거하고 새로 생성합니다. 사용자별 파일 lock으로 동시에 들어온 `join`/`leave`를 직렬화하며, 서로 다른 사용자의 명령은 독립적으로 실행합니다. `--join-timeout 10`으로 컨테이너 생성 이후의 인증 대기 시간을 지정할 수 있습니다. Docker image layer 생성에는 별도의 기본30초 timeout을 두므로 느린 VFS 복사가 짧은 인증 deadline을 소모하지 않습니다. 새 컨테이너를 만든 뒤 인증 거절, timeout, 검사 실패, Ctrl-C 또는 운영 CLI의 SIGTERM이 발생하면 그 컨테이너만 회수합니다. 기존 컨테이너의 경로가 다르거나 건강 확인이 실패하면 작업을 보존하고 명시적인 `leave`를 요청합니다.

`client alice run ...`은 `docker exec`로 별도의 CLI 프로세스와 socket 세션을 엽니다. `hold`의 연결을 공유하는 방식이 아닙니다. 두 연결은 Alice의 tenant quota와 권한을 공유하되 각 연결이 자신의 진행 중인 작업을 소유합니다. `run`은 결과를 기다린 뒤 연결을 닫고, `shell`은 여러 작업을 제출·조회·취소하는 동안 하나의 연결을 유지합니다. CLI의 전송·응답 처리 중 timeout이나 interrupt가 발생하면 연결을 닫아 늦게 도착한 응답이 다음 요청의 응답으로 섞이지 않게 합니다.

`leave`는 그 프로젝트의 해당 컨테이너를 정지하고 제거합니다. 모든 프로세스와 연결이 닫히므로 broker는 해당 연결들의 진행 중인 작업을 취소·정리합니다. 이미 나간 사용자에게 `leave`를 다시 실행해도 성공합니다. 사용자 등록과 token은 유지되므로 같은 명령으로 다시 참여할 수 있습니다. credential을 폐기하려면 호스트의 admin CLI에서 `admin revoke alice`를 실행합니다. 등록과 작업 기록까지 삭제해 이름과 등록 한도를 다시 쓰려면 연결·작업·GPU 자원 회수가 끝난 뒤 선택적으로 `admin forget alice`를 실행합니다. 아직 활성 상태이거나 회수 중이면 `tenant_busy`로 거절합니다. `forget`은 자동으로 실행되지 않으며 삭제한 이름을 다시 등록하면 새 credential이 발급되고 옛 credential은 사용할 수 없습니다.

```sh
python3 python/gpuctl.py --socket runtime/socket/gpu.sock --admin-file runtime/secrets/admin.token admin revoke alice
python3 python/gpuctl.py --socket runtime/socket/gpu.sock --admin-file runtime/secrets/admin.token admin inspect alice
python3 python/gpuctl.py --socket runtime/socket/gpu.sock --admin-file runtime/secrets/admin.token admin forget alice
```

컨테이너에는 nonroot UID/GID 1000, read-only root filesystem, capability 전부 제거, no-new-privileges, Docker 기본 seccomp, memory 128 MiB, swap 추가 사용 금지, PID 64개, CPU 0.5개, private IPC/cgroup namespace를 적용합니다. 호스트 경로는 dedicated socket 디렉토리와 자신의 token 파일 두 개만 mount할 수 있습니다.

## 실제 격리 확인

```sh
scripts/container-audit alice bob
```

`artifacts/container_isolation.json`에는 실행 중인 컨테이너에서 직접 읽은 결과가 저장됩니다.

- 호스트와 사용자 간, 사용자 서로 간 PID/mount/network/IPC namespace 분리
- 사용자별 `/tmp`에 쓴 파일의 다른 컨테이너 접근 차단
- UID/GID, capability 0, no-new-privileges, seccomp 활성화
- GPU device와 Docker socket 부재, 외부 네트워크 연결 차단
- 자신의 token만 전달되며 root/token 쓰기 차단
- read-only bind mount 목록과 실제 cgroup memory/PID/CPU 제한값

이 audit는 OS 격리의 증거입니다. 다른 사용자의 job 조회 거절과 GPU 결과 정확성은 broker 통합 테스트에서 별도로 확인합니다. 이 실습은 같은 host 계정에서 여러 사용자 환경을 운영하는 개발 실습이며, 서로 적대적인 사용자에게 host의 `scripts/container`나 rootless Docker daemon socket 접근 권한을 제공하는 배포 형태는 아닙니다. 운영자가 컨테이너 생성·자격증명 발급을 담당하고 사용자는 자기 컨테이너에만 접근하도록 해야 합니다.

전체 Docker/GPU 통합 검증은 `scripts/container-test`로 실행합니다. 컴파일된 `build/gpu-host`와 실행 중인 rootless Docker, build한 tenant image가 필요합니다. 테스트는 별도의 broker와 임의 suffix의 사용자 이름을 사용하여 기존 사용자 컨테이너에 영향을 주지 않습니다. 2개의 GPU lane에 3개의 idle 컨테이너를 인증하고 Alice/Bob의 작업을 동시에 실행합니다. CPU float/FMA 기준값과 GPU 결과, 다른 사용자의 GET 거절, SM mask의 분리, 실제 multi-job batch를 확인합니다. 다음에는 두 사용자가 GPU 작업 중일 때 Alice 컨테이너를 종료해 Alice 작업의 취소·메모리 반환을 확인합니다. Bob의 기존 장시간 작업이 계속 진행되고 같은 연결에서 새 작업도 정확히 완료되는지 확인한 뒤, Bob의 장시간 작업은 명시적으로 취소합니다. Docker stop 시간에 기대는 짧은 작업 대신 퇴장 완료 신호를 사용하므로 이 순서를 직접 검증할 수 있습니다. Charlie가 idle 연결을 유지하다가 반환된 lane을 사용해 작업을 완료하는지도 검사합니다.

이어 8회 가입·중복 가입·GPU 작업·퇴장·중복 퇴장을 반복하고, 매번 session·slot·lane 반환을 조회합니다. revoke에 따라 `hold`가 종료되는지, `forget` 후 같은 이름을 다시 등록할 수 있는지, 옛 credential이 거절되고 새 credential으로 재가입·GPU 작업이 성공하는지 검사합니다. broker shutdown 이후 남아 있는 `hold` 컨테이너의 자동 종료도 확인합니다. 이미지 안의 client SHA256과 현재 소스를 직접 대조하고, 테스트 중 관련 source와 broker executable이 바뀌지 않았는지도 확인합니다. 생성한 테스트 host·컨테이너·임시 token 디렉토리는 정리합니다. 기존 사용자 credential은 변경하지 않습니다. 결과는 `artifacts/container_integration.json`에 기록합니다.

CUDA나 Docker 없이 확인하는 회귀 테스트는 `python3 -m unittest discover -s tests -p test_client_protocol.py -v`와 `python3 -m unittest discover -s tests -p test_container_lifecycle.py -v`입니다. 전자는 실제 Unix socket을 사용해 응답 framing, timeout 후 연결 회수, `wait` deadline, revoke 감지, SIGTERM을 확인합니다. 후자는 Docker adapter를 대체하고 실제 파일 lock을 사용해 동시 가입·반복 퇴장·가입 실패 rollback·외부 프로젝트 보호를 확인합니다.

부정 입력 검증은 `scripts/container-negative-test`로 실행합니다. 공유 socket 폴더에 일반 파일 또는 token이 들어간 경우, mapped UID가 읽을 수 없는 token, 잘못된 사용자 이름, 다른 프로젝트 소유 컨테이너의 exec/leave를 거절하는지 검사합니다. 테스트는 임시 파일과 disposable Docker container만 사용합니다. cgroup controller 미지원 조건은 Docker info 응답을 mock해서 검증하므로 실제 host cgroup 설정을 변경하지 않습니다. 결과는 `artifacts/container_negative.json`에 기록합니다.

## 정리와 문제 확인

```sh
scripts/container leave alice
scripts/container leave bob
scripts/container daemon-stop
```

daemon 정지는 실행 중인 컨테이너가 있으면 거절합니다. 프로젝트 script가 시작한 rootlesskit process만 확인해서 정지하므로 다른 Docker daemon에 영향을 주지 않습니다. 로그는 `${XDG_CACHE_HOME:-~/.cache}/gpu-host-lab/docker/run/daemon.log`에 있습니다. 설치한 파일과 image는 재사용할 수 있도록 남습니다.

사용자 socket 디렉토리에 일반 파일이 있거나 token 파일이 읽기 불가능하면 `join`이 구체적인 오류를 냅니다. broker socket 이름은 `gpu.sock`을 사용합니다. daemon이 실행 중인데 API가 응답하지 않으면 새 daemon을 반복 실행하지 말고 기존 process와 로그를 확인합니다.

참고: [Docker rootless mode](https://docs.docker.com/engine/security/rootless/), [Docker static binary 설치](https://docs.docker.com/engine/install/binaries), [Docker run의 resource 및 isolation 옵션](https://docs.docker.com/engine/containers/run/).
