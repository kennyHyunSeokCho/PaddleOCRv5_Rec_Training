## 한 파일로 실행하는 PP-OCRv5 한국어 인식 학습 파이프라인

이 폴더는 `.env`로 경로/파라미터를 관리하고, 단일 커맨드로 학습을 시작할 수 있게 구성되어 있습니다.

### 구성 파일
- `setup_conda.sh`: Conda/Miniconda 자동 설치, GPU 자동 감지, Paddle(GPU/CPU) 설치, 설치 검증까지 수행
- `.env.example`: 필요한 경로/파라미터 예시. 복사하여 `.env`로 수정하세요
- `env/recognition.env.example`: 주석이 풍부한 인식 모델용 템플릿. 필요 키를 골라 `.env`로 옮겨 사용
- `run_train.py`: `.env`를 로드하여 `tools/train.py`를 적절히 호출 (단일 GPU 강제, tqdm 유지, 클라우드 경로 보정/자동 클론 지원)
- `train.sh`: Conda 활성화 후 `run_train.py`를 실행하는 단일 진입점

### Conda/GPU 환경 준비
```bash
cd /Users/johyeonseog/Downloads/PaddleOCR/training_pipeline_rec
# 기본 (GPU 자동 감지, 없으면 CPU)
./setup_conda.sh
# GPU 강제/버전 선택
USE_GPU=true PADDLE_CUDA_INDEX=cu121 ./setup_conda.sh
# 환경 이름/파이썬 버전 지정
ENV_NAME=ocr_rec PYTHON_VERSION=3.10 ./setup_conda.sh
```
- 미설치 시 Miniconda를 자동 설치하고, 이후 conda 환경을 생성/활성합니다.
- `nvidia-smi` 감지 시 GPU 버전(`paddlepaddle-gpu`)을 설치하며, 인덱스는 `PADDLE_CUDA_INDEX`로 제어합니다(`cu118|cu121|cu126`).
- 마지막에 Paddle 간단 연산으로 설치를 검증합니다.

### env 템플릿 사용법
1) 템플릿을 복사해 시작:
```bash
cp env/recognition.env.example .env
# 경로/옵션을 절대경로 기준으로 수정
```
2) 최소 필수 키(예):
```
REPO_DIR=/workspace/PaddleOCR
BASE_CONFIG=/workspace/PaddleOCR/configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml
DATA_DIR=/data/ocr_rec_dataset
TRAIN_LIST=/data/ocr_rec_dataset/train.txt
VAL_LIST=/data/ocr_rec_dataset/val.txt
SAVE_DIR=/workspace/PaddleOCR/output/rec_korean_v5
GPUS=0
```
3) 실행
```bash
./train.sh /Users/johyeonseog/Downloads/PaddleOCR/training_pipeline_rec/training_config.env
```

### 자동 저장소 클론(옵션)
```
AUTO_CLONE_REPO=true
REPO_GIT_URL=https://github.com/PaddlePaddle/PaddleOCR.git
```

### 데이터 구조/클라우드 경로 보정/로그 설정
- 본문 상단 ‘데이터 구조 가이드’, ‘클라우드/마운트 경로 보정’, ‘tqdm 및 실행 로그’를 참고하세요.
