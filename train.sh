#!/bin/bash
# 단일 진입점: Conda 환경 로드 → run_train.py 실행
# 사용법:
#   ./train.sh                             # ./training_pipeline_rec/.env 사용
#   ./train.sh /path/to/.env               # 지정한 .env 사용
#   ./train.sh /path/to/.env --epochs 100 --batch-size 64 --gpus 0 --use-amp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 첫 인자가 .env 파일이면 소비하고, 나머지는 run_train.py에 전달
ENV_FILE="${1:-${SCRIPT_DIR}/training_config.env}"
EXTRA_ARGS=()
if [[ -f "${ENV_FILE}" ]]; then
  shift || true
  EXTRA_ARGS=("$@")
else
  # 첫 인자가 .env가 아니면 기본 training_config.env 사용하고, 모든 인자를 그대로 전달
  ENV_FILE="${SCRIPT_DIR}/training_config.env"
  EXTRA_ARGS=("$@")
fi

if [ ! -f "${ENV_FILE}" ]; then
  echo "[오류] .env 파일을 찾을 수 없습니다: ${ENV_FILE}" >&2
  exit 1
fi

# Conda 초기화 및 활성화
if ! command -v conda >/dev/null 2>&1; then
  # Miniconda가 설치되어 있으면 PATH에 추가
  if [ -d "${SCRIPT_DIR}/miniconda3" ]; then
    export PATH="${SCRIPT_DIR}/miniconda3/bin:$PATH"
    source "${SCRIPT_DIR}/miniconda3/etc/profile.d/conda.sh"
    echo "[정보] Miniconda PATH 설정 완료"
  elif [ -d "/content/PaddleOCRv5_Rec_Training/miniconda3" ]; then
    export PATH="/content/PaddleOCRv5_Rec_Training/miniconda3/bin:$PATH"
    source "/content/PaddleOCRv5_Rec_Training/miniconda3/etc/profile.d/conda.sh"
    echo "[정보] Miniconda PATH 설정 완료 (주피터 환경)"
  else
    echo "[오류] conda 명령을 찾을 수 없습니다. 먼저 설치/초기화하세요." >&2
    exit 1
  fi
fi

eval "$(conda shell.bash hook)"
ENV_NAME_VAL="${ENV_NAME:-ocr_paddle}"
conda activate "${ENV_NAME_VAL}" || {
  echo "[경고] '${ENV_NAME_VAL}' 환경을 활성화하지 못했습니다. setup_conda.sh를 먼저 실행하세요." >&2
  exit 1
}

python "${SCRIPT_DIR}/run_train.py" --env "${ENV_FILE}" ${EXTRA_ARGS[@]:-}
