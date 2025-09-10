#!/bin/zsh
# macOS/Linux GPU 지원 Conda 환경 구성 스크립트
# - Conda 미설치 시 Miniconda 자동 설치
# - GPU 환경 자동 감지(nvidia-smi) 후 paddlepaddle-gpu 또는 CPU 패키지 설치
# - 설치 검증 수행
# 사용법:
#   ./setup_conda.sh                           # 기본값(ENV_NAME=ocr_paddle, PYTHON_VERSION=3.10)
#   ENV_NAME=ocr_rec PYTHON_VERSION=3.10 ./setup_conda.sh
#   USE_GPU=false PADDLE_CUDA_INDEX=cu118 ./setup_conda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 기본 파라미터
ENV_NAME="${ENV_NAME:-ocr_paddle}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PADDLE_VERSION="${PADDLE_VERSION:-3.1.0}"
MINICONDA_DIR="${MINICONDA_DIR:-${SCRIPT_DIR}/miniconda3}"

# GPU 감지 (nvidia-smi) 또는 강제 설정
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPU=true
else
  DETECTED_GPU=false
fi
USE_GPU="${USE_GPU:-${DETECTED_GPU}}"
# CUDA 인덱스(리눅스/NVIDIA): cu118 | cu121 | cu126
PADDLE_CUDA_INDEX="${PADDLE_CUDA_INDEX:-cu121}"

print_step() {
  echo "\n========== $1 =========="
}

fail() {
  echo "[오류] $1" >&2
  exit 1
}

ensure_miniconda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  print_step "Miniconda 설치 준비"
  # OS/ARCH 감지
  OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
  ARCH_NAME="$(uname -m)"
  INSTALLER_URL=""
  if [[ "${OS_NAME}" == "darwin" ]]; then
    if [[ "${ARCH_NAME}" == "arm64" ]]; then
      INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
      INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
  elif [[ "${OS_NAME}" == "linux" ]]; then
    if [[ "${ARCH_NAME}" == "aarch64" ]]; then
      INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    else
      INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
  else
    fail "지원되지 않는 OS: ${OS_NAME}"
  fi

  mkdir -p "${MINICONDA_DIR%/*}"
  TMP_SH="${SCRIPT_DIR}/miniconda_installer.sh"
  echo "[다운로드] ${INSTALLER_URL}"
  curl -LfsS -o "${TMP_SH}" "${INSTALLER_URL}" || fail "Miniconda 다운로드 실패"
  chmod +x "${TMP_SH}"
  echo "[설치] Miniconda -> ${MINICONDA_DIR}"
  bash "${TMP_SH}" -b -p "${MINICONDA_DIR}" || fail "Miniconda 설치 실패"
  rm -f "${TMP_SH}"
  # conda 초기화
  if [ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
    . "${MINICONDA_DIR}/etc/profile.d/conda.sh"
  fi
  conda --version || fail "conda 초기화 실패"
}

ensure_conda_initialized() {
  if command -v conda >/dev/null 2>&1; then
    # shell hook
    if command -v zsh >/dev/null 2>&1; then
      eval "$(conda shell.zsh hook)"
    else
      eval "$(conda shell.bash hook)"
    fi
    return 0
  fi
  ensure_miniconda
  if command -v zsh >/dev/null 2>&1; then
    . "${MINICONDA_DIR}/etc/profile.d/conda.sh" || true
    eval "$(conda shell.zsh hook)"
  else
    . "${MINICONDA_DIR}/etc/profile.d/conda.sh" || true
    eval "$(conda shell.bash hook)"
  fi
}

create_env() {
  print_step "Conda 환경 생성/활성: ${ENV_NAME} (Python ${PYTHON_VERSION})"
  
  # TOS 동의 (필요한 경우)
  echo "[TOS] Conda Terms of Service 동의 처리"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  
  if conda env list | grep -E "^${ENV_NAME}\s" >/dev/null; then
    echo "[정보] 기존 환경 발견: ${ENV_NAME}"
  else
    # conda-forge 채널 사용으로 TOS 문제 회피
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -c conda-forge || fail "환경 생성 실패"
  fi
  conda activate "${ENV_NAME}" || fail "환경 활성화 실패"
  python -V
}

install_requirements() {
  print_step "pip 업그레이드 및 필수 패키지 설치"
  python -m pip install -U pip wheel setuptools || fail "pip 업그레이드 실패"
  pip install tqdm python-dotenv pyyaml scikit-image opencv-python pillow albumentations lmdb shapely pyclipper || fail "유틸 설치 실패"
  if [ -f "${REPO_DIR}/requirements.txt" ]; then
    echo "[설치] repo requirements.txt"
    pip install -r "${REPO_DIR}/requirements.txt" || fail "requirements 설치 실패"
  fi
}

install_paddle() {
  if [[ "${USE_GPU}" == "true" ]]; then
    print_step "PaddlePaddle GPU 설치 (index: ${PADDLE_CUDA_INDEX}, ver: ${PADDLE_VERSION})"
    # 우선 지정된 인덱스로 설치, 실패 시 다른 인덱스 순차 시도
    indexes=("${PADDLE_CUDA_INDEX}")
    # 보조 후보
    if [[ "${PADDLE_CUDA_INDEX}" == "cu126" ]]; then
      indexes+=("cu121" "cu118")
    elif [[ "${PADDLE_CUDA_INDEX}" == "cu121" ]]; then
      indexes+=("cu118" "cu126")
    else
      indexes+=("cu121" "cu126")
    fi
    local ok=0
    for idx in "${indexes[@]}"; do
      echo "[try] idx=${idx}"
      if pip install "paddlepaddle-gpu==${PADDLE_VERSION}" -i "https://www.paddlepaddle.org.cn/packages/stable/${idx}/"; then
        ok=1; break
      fi
    done
    if [[ ${ok} -ne 1 ]]; then
      fail "paddlepaddle-gpu 설치 실패"
    fi
  else
    print_step "PaddlePaddle CPU 설치 (ver: ${PADDLE_VERSION})"
    pip install "paddlepaddle==${PADDLE_VERSION}" || fail "paddlepaddle 설치 실패"
  fi
}

verify_install() {
  print_step "설치 검증"
  python - <<'PY'
import paddle, time
print('paddle version:', paddle.__version__)
try:
    is_cuda = paddle.is_compiled_with_cuda()
except Exception:
    is_cuda = False
print('compiled with CUDA?:', is_cuda)
try:
    dev = 'gpu' if is_cuda else 'cpu'
    paddle.set_device(dev)
    print('current device:', paddle.device.get_device())
    x = paddle.randn([1024, 1024])
    y = paddle.matmul(x, x.t())
    print('matmul OK on:', y.place)
except Exception as e:
    print('GPU/CPU 테스트 중 예외:', repr(e))
PY
}

# 실행 흐름
print_step "Conda 확인/초기화"
if ! command -v conda >/dev/null 2>&1; then
  ensure_miniconda
fi
ensure_conda_initialized
create_env
install_requirements
install_paddle
verify_install

echo "\n[완료] Conda 환경 준비 완료: ${ENV_NAME}"
echo "[힌트] conda activate ${ENV_NAME} && python ${SCRIPT_DIR}/run_train.py --env ${SCRIPT_DIR}/training_config.env"
