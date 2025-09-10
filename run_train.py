#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
한국어 PP-OCRv5 모바일 인식 학습 실행 스크립트
- .env 파일을 로드하여 경로/하이퍼파라미터를 설정하고
- PaddleOCR의 tools/train.py 를 적절한 override와 함께 실행합니다.

사용법:
  python run_train.py --env ./.env
  # 또는 .env + 일부 인자 CLI로 재정의
  python run_train.py --env ./.env --epochs 100 --batch-size 64 --gpus 0 \
    --data-dir /data/ocr --train-list /data/ocr/train.txt --val-list /data/ocr/val.txt \
    --save-dir /workspace/PaddleOCR/output/exp1 --use-amp

주의:
  - 경로는 절대경로 사용 권장
  - 단일 GPU만 사용하며, 여러 GPU가 지정되어도 첫 번째 GPU만 사용합니다.
  - tqdm 진행률이 그대로 보이도록 하위 프로세스의 stdout/stderr를 캡처하지 않습니다.
  - 클라우드 GPU에서 로컬/마운트 경로 차이를 보정하기 위한 라벨 경로 재기록 기능을 제공합니다.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from argparse import BooleanOptionalAction
except Exception:
    # Python < 3.9 호환: 대체 불리언 플래그
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            super().__init__(option_strings, dest, nargs=0, default=default, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

# 간단한 .env 로더 (python-dotenv 없이 동작)
# - 주석(#)과 공백 라인을 무시
# - KEY=VALUE 형식만 지원
# - 작은따옴표/큰따옴표는 양끝에 있을 경우 제거

def load_env_file(env_path: Path) -> Dict[str, str]:
    """.env 파일을 파싱하여 dict로 반환"""
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_path}")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in ['"', "'"]):
            value = value[1:-1]
        env_vars[key] = value
    return env_vars


def to_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def build_overrides(
    repo_dir: Path,
    base_config: Path,
    save_dir: Path,
    data_dir: Path,
    train_list: Path,
    val_list: Path,
    epochs: int,
    batch_size: int,
    gpus: str,
    use_amp: bool,
    pretrain: str,
    checkpoints: str,
    hard_list: str,
    hard_ratio: float,
) -> List[str]:
    """tools.train.py -o 에 전달할 기본 override 리스트 구성"""
    # 단일 GPU만 사용 (분산 비활성화)
    is_distributed = False

    overrides: List[str] = [
        f"Global.save_model_dir={str(save_dir)}",
        f"Train.dataset.data_dir={str(data_dir)}",
        f"Eval.dataset.data_dir={str(data_dir)}",
        f"Global.epoch_num={int(epochs)}",
        f"Train.loader.batch_size_per_card={int(batch_size)}",
        f"Eval.loader.batch_size_per_card={int(batch_size)}",  # 평가용 배치 사이즈도 동일하게 설정
        f"Train.sampler.first_bs={int(batch_size)}",  # MultiScaleSampler의 first_bs 설정
        f"Global.distributed={'True' if is_distributed else 'False'}",
        f"Global.use_gpu={'True' if (gpus and gpus.strip()) else 'False'}",
    ]

    # 라벨 파일 설정 (하드샘플 포함 여부)
    if hard_list and hard_ratio > 0:
        overrides += [
            f"Train.dataset.label_file_list={[str(train_list), str(hard_list)]}",
            f"Train.dataset.ratio_list={[1.0, float(hard_ratio)]}",
            f"Eval.dataset.label_file_list={[str(val_list)]}",
        ]
    else:
        overrides += [
            f"Train.dataset.label_file_list={[str(train_list)]}",
            f"Eval.dataset.label_file_list={[str(val_list)]}",
        ]

    if use_amp:
        overrides.append("Global.use_amp=True")

    # 프리트레인/체크포인트
    if pretrain:
        overrides.append(f"Global.pretrained_model={pretrain}")
    overrides.append(f"Global.checkpoints={checkpoints}")  # 비어있으면 이어달리기 없음

    return overrides


def _parse_label_line(line: str) -> Tuple[str, str]:
    """라벨 파일 한 줄 파싱: 탭 우선, 없으면 첫 공백 분리"""
    line = line.strip()
    if "\t" in line:
        img_path, label = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            raise ValueError("잘못된 라벨 형식: 공백/탭으로 분리된 라벨이 없습니다")
        img_path, label = parts[0], parts[1]
    return img_path, label


def quick_check_labels(label_file: Path, data_dir: Path, max_check: int = 50) -> Tuple[int, int, List[str]]:
    """라벨 파일 샘플 검증: 이미지 경로 존재여부 확인
    반환: (검사한 라인 수, 존재하는 이미지 수, 예시 3개)
    """
    checked = 0
    ok = 0
    examples: List[str] = []
    if not label_file.exists():
        raise FileNotFoundError(f"라벨 파일 없음: {label_file}")
    with label_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                img_rel, label = _parse_label_line(line)
            except Exception:
                continue
            img_path = Path(img_rel)
            if not img_path.is_absolute():
                img_path = (data_dir / img_rel)
            if img_path.exists():
                ok += 1
                if len(examples) < 3:
                    examples.append(f"{img_path} -> {label[:18]}...")
            checked += 1
            if checked >= max_check:
                break
    return checked, ok, examples


def download_example_data(data_dir: Path) -> Tuple[Path, Path]:
    """PaddleOCR 공식 예시 데이터 다운로드"""
    import urllib.request
    import tarfile
    import tempfile
    
    url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar"
    
    print(f"[다운로드] PaddleOCR 공식 예시 데이터 다운로드 중...")
    print(f"[URL] {url}")
    
    try:
        # 데이터 디렉토리 생성
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 파일로 다운로드
        with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp_file:
            print("[다운로드] 파일 다운로드 중...")
            urllib.request.urlretrieve(url, tmp_file.name)
            
            # tar 파일 압축 해제
            print("[압축해제] 데이터 압축 해제 중...")
            with tarfile.open(tmp_file.name, 'r') as tar:
                tar.extractall(data_dir)
            
            # 임시 파일 삭제
            import os
            os.unlink(tmp_file.name)
        
        # 압축 해제된 데이터 확인 (ocr_rec_dataset_examples 구조)
        example_dir = data_dir / "ocr_rec_dataset_examples"
        if example_dir.exists():
            # ocr_rec_dataset_examples 구조에서 train.txt, val.txt 찾기
            train_path = example_dir / "train.txt"
            val_path = example_dir / "val.txt"
            
            if train_path.exists() and val_path.exists():
                print(f"[완료] Recognition 예시 데이터 다운로드 완료:")
                print(f"  - 학습 데이터: {train_path}")
                print(f"  - 검증 데이터: {val_path}")
                return train_path, val_path
            else:
                print(f"[경고] {example_dir}에서 train.txt/val.txt를 찾을 수 없습니다")
                return None, None
        else:
            print(f"[경고] 압축 해제 후 {example_dir} 디렉토리가 생성되지 않았습니다")
            return None, None
            
    except Exception as e:
        print(f"[오류] 예시 데이터 다운로드 실패: {e}")
        return None, None

def discover_dataset(data_dir: Path) -> Tuple[Path, Path, Path]:
    """DATA_DIR 하에서 표준 라벨 파일을 자동 탐색
    우선순위: train.txt/val.txt → rec_gt_train.txt/rec_gt_test.txt → 하위 디렉토리(1~2뎁스)
    반환: (data_dir, train_list, val_list)
    """
    def _pair_in(dir_path: Path) -> Tuple[Path, Path] | None:
        cands = [
            (dir_path / "train.txt", dir_path / "val.txt"),
            (dir_path / "rec_gt_train.txt", dir_path / "rec_gt_test.txt"),
        ]
        for tr, va in cands:
            if tr.exists() and va.exists():
                return tr, va
        return None

    # 0) 직접 매치
    found = _pair_in(data_dir)
    if found:
        return data_dir, found[0], found[1]

    # 1) ocr_rec_dataset_examples 우선
    sub = data_dir / "ocr_rec_dataset_examples"
    if sub.exists():
        found = _pair_in(sub)
        if found:
            return sub, found[0], found[1]

    # 2) 1~2 depth 탐색 (디렉토리가 존재할 때만)
    if data_dir.exists():
        for p in data_dir.iterdir():
            if p.is_dir():
                found = _pair_in(p)
                if found:
                    return p, found[0], found[1]
                # 2뎁스
                for q in p.iterdir():
                    if q.is_dir():
                        found = _pair_in(q)
                        if found:
                            return q, found[0], found[1]

    # 실패 시 기존 경로 반환
    return data_dir, data_dir / "train.txt", data_dir / "val.txt"


def rewrite_labels(
    label_in: Path,
    dst_dir: Path,
    data_dir: Path,
    from_prefix: str | None,
    to_prefix: str | None,
    make_relative: bool,
) -> Path:
    """라벨 파일의 이미지 경로를 재기록하여 새 라벨 파일을 생성
    - from_prefix/to_prefix: 절대 경로 접두사 바꾸기 (클라우드 마운트 경로 치환)
    - make_relative: 최종적으로 data_dir 기준 상대경로로 변환
    반환: 새로 생성된 라벨 파일 경로
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / label_in.name
    from_prefix = (from_prefix or "").strip()
    to_prefix = (to_prefix or "").strip()

    rewritten = 0
    total = 0
    with label_in.open("r", encoding="utf-8") as fi, out_path.open("w", encoding="utf-8") as fo:
        for raw in fi:
            line = raw.rstrip("\n")
            if not line.strip():
                fo.write(raw)
                continue
            try:
                img_str, label = _parse_label_line(line)
            except Exception:
                fo.write(raw + "\n" if not raw.endswith("\n") else raw)
                continue

            img_path = Path(img_str)
            # 1) prefix rewrite for absolute path
            if from_prefix and to_prefix and img_path.is_absolute():
                img_str_mod = str(img_path)
                if img_str_mod.startswith(from_prefix):
                    img_str_mod = to_prefix + img_str_mod[len(from_prefix) :]
                    img_path = Path(img_str_mod)
                    rewritten += 1

            # 2) to relative path under data_dir
            if make_relative:
                if img_path.is_absolute():
                    try:
                        img_path = img_path.relative_to(data_dir)
                    except Exception:
                        # 절대경로가 data_dir 바깥이면 to_prefix로 먼저 끌어들이기 필요
                        if to_prefix and str(img_path).startswith(to_prefix):
                            try:
                                img_path = img_path.relative_to(data_dir)
                            except Exception:
                                pass
                else:
                    # 이미 상대경로면 그대로 둠
                    pass

            # 3) 최종 출력 (탭 구분 유지)
            fo.write(f"{img_path}\t{label}\n")
            total += 1

    print(f"[rewrite] {label_in.name}: rewritten={rewritten}, total={total} -> {out_path}")
    return out_path


def main() -> int:
    # 인자 파싱
    parser = argparse.ArgumentParser(description="PP-OCRv5 Korean Mobile Recognition Trainer")
    parser.add_argument("--env", type=str, default=".env", help=".env 파일 경로")
    # 경로/환경 인자
    parser.add_argument("--repo-dir", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--train-list", type=str, default=None)
    parser.add_argument("--val-list", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    # 학습 하이퍼파라미터/옵션
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpus", type=str, default=None, help="단일 GPU ID (예: 0). 여러 개여도 첫 번째만 사용")
    parser.add_argument("--use-amp", action=BooleanOptionalAction, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default=None)
    # 데이터/라벨 자동탐색 및 경로 재기록
    parser.add_argument("--auto-discover-data", action=BooleanOptionalAction, default=None)
    parser.add_argument("--download-example-data", action=BooleanOptionalAction, default=None, help="PaddleOCR 공식 예시 데이터 자동 다운로드")
    parser.add_argument("--path-rewrite-from", type=str, default=None)
    parser.add_argument("--path-rewrite-to", type=str, default=None)
    parser.add_argument("--auto-rewrite-to-relative", action=BooleanOptionalAction, default=None)
    # git 자동 클론/업데이트
    parser.add_argument("--auto-clone-repo", action=BooleanOptionalAction, default=None)
    parser.add_argument("--repo-git-url", type=str, default=None)
    parser.add_argument("--repo-git-update", action=BooleanOptionalAction, default=None)
    parser.add_argument("--repo-branch", type=str, default=None)
    parser.add_argument("--repo-ref", type=str, default=None)
    # 로깅/평가/로더 세부
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-batch-step", type=int, default=None)
    parser.add_argument("--save-epoch-step", type=int, default=None)
    parser.add_argument("--eval-batch-step", type=str, default=None, help="정수 또는 start,step")
    parser.add_argument("--eval-batch-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-wandb", action=BooleanOptionalAction, default=None)
    # 임의 오버라이드
    parser.add_argument("-o", "--override", action="append", default=None, help="추가 override, 예: Global.log_smooth_window=20")

    args = parser.parse_args()

    env_path = Path(args.env).expanduser().resolve()
    env = load_env_file(env_path)

    # 필수 경로 로드 및 기본값 보정 (.env)
    repo_dir = Path(env.get("REPO_DIR", ".")).expanduser().resolve()
    base_config = Path(env.get("BASE_CONFIG", "")).expanduser().resolve()
    data_dir = Path(env.get("DATA_DIR", "")).expanduser().resolve()
    train_list = Path(env.get("TRAIN_LIST", "")).expanduser().resolve()
    val_list = Path(env.get("VAL_LIST", "")).expanduser().resolve()
    save_dir = Path(env.get("SAVE_DIR", repo_dir / "output/rec_korean_v5")).expanduser().resolve()

    # CLI 인자 우선 적용 (경로류)
    if args.repo_dir: repo_dir = Path(args.repo_dir).expanduser().resolve()
    if args.base_config: base_config = Path(args.base_config).expanduser().resolve()
    if args.data_dir: data_dir = Path(args.data_dir).expanduser().resolve()
    if args.train_list: train_list = Path(args.train_list).expanduser().resolve()
    if args.val_list: val_list = Path(args.val_list).expanduser().resolve()
    if args.save_dir: save_dir = Path(args.save_dir).expanduser().resolve()

    # 자동 클론/업데이트 옵션 (기본: 자동 클론 활성)
    auto_clone = to_bool(env.get("AUTO_CLONE_REPO", "true"))
    repo_git_url = env.get("REPO_GIT_URL", "https://github.com/PaddlePaddle/PaddleOCR.git").strip()
    repo_git_update = to_bool(env.get("REPO_GIT_UPDATE", "false"))
    repo_branch = env.get("REPO_BRANCH", "").strip()
    repo_ref = env.get("REPO_REF", "").strip()
    # CLI 우선
    if args.auto_clone_repo is not None: auto_clone = args.auto_clone_repo
    if args.repo_git_url: repo_git_url = args.repo_git_url
    if args.repo_git_update is not None: repo_git_update = args.repo_git_update
    if args.repo_branch: repo_branch = args.repo_branch
    if args.repo_ref: repo_ref = args.repo_ref

    if auto_clone and not repo_dir.exists():
        print(f"[clone] {repo_git_url} -> {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_git_url, str(repo_dir)], check=True)

    if repo_dir.exists() and repo_git_update:
        print(f"[git] fetch/pull in {repo_dir}")
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--prune"], check=True)
        if repo_ref:
            subprocess.run(["git", "-C", str(repo_dir), "checkout", repo_ref], check=True)
        elif repo_branch:
            subprocess.run(["git", "-C", str(repo_dir), "checkout", repo_branch], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
        else:
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)

    # 하이퍼파라미터/옵션 (.env)
    epochs = int(env.get("EPOCHS", "20"))
    batch_size = int(env.get("BATCH_SIZE", "128"))
    use_amp = to_bool(env.get("USE_AMP", "false"))
    gpus_raw = env.get("GPUS", "").strip()
    pretrain = env.get("PRETRAIN", "")
    checkpoints = env.get("CHECKPOINTS", "")
    hard_list = env.get("HARD_LIST", "")
    hard_ratio = float(env.get("HARD_RATIO", "0"))

    # CLI 우선 (학습)
    if args.epochs is not None: epochs = args.epochs
    if args.batch_size is not None: batch_size = args.batch_size
    if args.use_amp is not None: use_amp = args.use_amp
    if args.gpus is not None: gpus_raw = args.gpus
    if args.pretrain is not None: pretrain = args.pretrain
    if args.checkpoints is not None: checkpoints = args.checkpoints

    # 추가 .env 파라미터(선택): seed/로깅/평가/로더
    seed_val = env.get("SEED", "").strip()
    print_batch_step = env.get("PRINT_BATCH_STEP", "").strip()
    save_epoch_step = env.get("SAVE_EPOCH_STEP", "").strip()
    eval_batch_step_raw = env.get("EVAL_BATCH_STEP", "").strip()  # "2000" 또는 "0,2000"
    eval_batch_epoch = env.get("EVAL_BATCH_EPOCH", "").strip()
    num_workers = env.get("NUM_WORKERS", "").strip()
    use_wandb = env.get("USE_WANDB", "").strip()

    # CLI 우선 (세부)
    if args.seed is not None: seed_val = str(args.seed)
    if args.print_batch_step is not None: print_batch_step = str(args.print_batch_step)
    if args.save_epoch_step is not None: save_epoch_step = str(args.save_epoch_step)
    if args.eval_batch_step is not None: eval_batch_step_raw = args.eval_batch_step
    if args.eval_batch_epoch is not None: eval_batch_epoch = str(args.eval_batch_epoch)
    if args.num_workers is not None: num_workers = str(args.num_workers)
    if args.use_wandb is not None: use_wandb = "true" if args.use_wandb else "false"

    # 클라우드/마운트 경로 보정 관련 옵션
    auto_discover = to_bool(env.get("AUTO_DISCOVER_DATA", "true"))
    download_example = to_bool(env.get("DOWNLOAD_EXAMPLE_DATA", "false"))
    path_rewrite_from = env.get("PATH_REWRITE_FROM", "")
    path_rewrite_to = env.get("PATH_REWRITE_TO", "")
    make_relative = to_bool(env.get("AUTO_REWRITE_TO_RELATIVE", "false"))
    # CLI 우선
    if args.auto_discover_data is not None: auto_discover = args.auto_discover_data
    if args.download_example_data is not None: download_example = args.download_example_data
    if args.path_rewrite_from is not None: path_rewrite_from = args.path_rewrite_from
    if args.path_rewrite_to is not None: path_rewrite_to = args.path_rewrite_to
    if args.auto_rewrite_to_relative is not None: make_relative = args.auto_rewrite_to_relative

    # 단일 GPU 강제: 여러 개 지정되면 첫 번째만 사용
    selected_gpu = ""
    if gpus_raw:
        parts = [p.strip() for p in gpus_raw.split(",") if p.strip()]
        if parts:
            selected_gpu = parts[0]
            if len(parts) > 1:
                print(f"[info] 여러 GPU가 지정되었지만 단일 GPU만 사용합니다: {selected_gpu}")

    # 경로 검증 (repo/config)
    for p in [repo_dir, base_config]:
        if not Path(p).exists():
            raise FileNotFoundError(f"경로를 확인하세요: {p}")

    # 데이터 준비 (다운로드 또는 탐색)
    # 예시 데이터 다운로드 (우선순위 높음)
    if download_example:
        print("[예시 데이터] PaddleOCR 공식 예시 데이터 다운로드 중...")
        example_train, example_val = download_example_data(data_dir)
        if example_train and example_val:
            train_list = example_train
            val_list = example_val
            data_dir = train_list.parent  # 데이터 디렉토리 업데이트
            print(f"[예시 데이터] 다운로드 완료: {train_list}, {val_list}")
        else:
            print("[예시 데이터] 다운로드 실패, 기존 데이터 사용")
    
    # 데이터셋 자동 탐색 (예시 데이터가 없을 때만)
    elif auto_discover and (not train_list.exists() or not val_list.exists()):
        new_root, tr_auto, va_auto = discover_dataset(data_dir)
        print(f"[discover] data_dir={new_root}\n  train={tr_auto}\n  val={va_auto}")
        data_dir = new_root
        train_list = tr_auto
        val_list = va_auto

    # DATA_DIR 존재 확인 (데이터 준비 후)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR이 존재하지 않습니다: {data_dir}")

    # 라벨/이미지 빠른 검증
    tr_checked, tr_ok, tr_ex = quick_check_labels(train_list, data_dir)
    va_checked, va_ok, va_ex = quick_check_labels(val_list, data_dir)
    print(f"[dataset] train checked: {tr_checked}, exist: {tr_ok}")
    for e in tr_ex:
        print(f"  [ex] {e}")
    print(f"[dataset] val   checked: {va_checked}, exist: {va_ok}")
    for e in va_ex:
        print(f"  [ex] {e}")

    # 경로 치환/상대경로화가 필요한 경우 라벨 재기록
    need_rewrite = False
    if path_rewrite_from and path_rewrite_to:
        print(f"[rewrite-plan] FROM='{path_rewrite_from}' TO='{path_rewrite_to}'")
        need_rewrite = True
    if make_relative:
        print(f"[rewrite-plan] 라벨 이미지 경로를 data_dir 상대경로로 변환")
        need_rewrite = True

    if need_rewrite:
        rewrite_dir = save_dir / "rewritten_labels"
        train_list = rewrite_labels(
            label_in=train_list,
            dst_dir=rewrite_dir,
            data_dir=data_dir,
            from_prefix=path_rewrite_from,
            to_prefix=path_rewrite_to,
            make_relative=make_relative,
        )
        val_list = rewrite_labels(
            label_in=val_list,
            dst_dir=rewrite_dir,
            data_dir=data_dir,
            from_prefix=path_rewrite_from,
            to_prefix=path_rewrite_to,
            make_relative=make_relative,
        )
        # 재검증 로그
        tr_checked, tr_ok, _ = quick_check_labels(train_list, data_dir)
        va_checked, va_ok, _ = quick_check_labels(val_list, data_dir)
        print(f"[dataset:rewritten] train exist: {tr_ok}/{tr_checked}, val exist: {va_ok}/{va_checked}")
        if tr_ok == 0:
            print("[경고] 재기록 이후에도 유효한 학습 이미지가 없습니다. PATH_REWRITE_* 또는 DATA_DIR를 확인하세요.")

    # 항상 비분산 실행 (단일 GPU 또는 CPU)
    cmd: List[str] = [sys.executable]
    if selected_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu

    overrides = build_overrides(
        repo_dir=repo_dir,
        base_config=base_config,
        save_dir=save_dir,
        data_dir=data_dir,
        train_list=train_list,
        val_list=val_list,
        epochs=epochs,
        batch_size=batch_size,
        gpus=selected_gpu,
        use_amp=use_amp,
        pretrain=pretrain,
        checkpoints=checkpoints,
        hard_list=hard_list,
        hard_ratio=hard_ratio,
    )

    # ===== 추가 .env/CLI 기반 오버라이드 =====
    # SEED
    if seed_val:
        try:
            _ = int(seed_val)
            overrides.append(f"Global.seed={int(seed_val)}")
        except Exception:
            pass
    # PRINT/SAVE/EVAL 스텝
    if print_batch_step:
        try:
            overrides.append(f"Global.print_batch_step={int(print_batch_step)}")
        except Exception:
            pass
    if save_epoch_step:
        try:
            overrides.append(f"Global.save_epoch_step={int(save_epoch_step)}")
        except Exception:
            pass
    if eval_batch_step_raw:
        # 형태: "2000" 또는 "0,2000"
        if "," in eval_batch_step_raw:
            parts = [p.strip() for p in eval_batch_step_raw.split(",") if p.strip()]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                overrides.append(f"Global.eval_batch_step={[int(parts[0]), int(parts[1]) ]}")
        else:
            if eval_batch_step_raw.isdigit():
                overrides.append(f"Global.eval_batch_step={int(eval_batch_step_raw)}")
    if eval_batch_epoch:
        try:
            overrides.append(f"Global.eval_batch_epoch={int(eval_batch_epoch)}")
        except Exception:
            pass
    # 로더 workers
    if num_workers:
        try:
            overrides.append(f"Train.loader.num_workers={int(num_workers)}")
        except Exception:
            pass
    # W&B 사용 플래그
    if use_wandb and to_bool(use_wandb):
        overrides.append("Global.use_wandb=True")

    # 임의의 OVERRIDE_* 환경변수 패스스루
    for k, v in env.items():
        if k.startswith("OVERRIDE_") and v:
            overrides.append(v)
    # CLI --override KEY=YAML (여러 번)
    if args.override:
        for ov in args.override:
            if ov and "=" in ov:
                overrides.append(ov)

    # 최종 커맨드
    cmd += [
        str(repo_dir / "tools" / "train.py"),
        "-c",
        str(base_config),
        "-o",
        *overrides,
    ]

    # 실시간 로그 직렬화 설정 (tqdm 살리기 위해 stdout/stderr 캡처하지 않음)
    env_vars = os.environ.copy()
    env_vars["PYTHONUNBUFFERED"] = "1"
    env_vars["FLAGS_logging_serialize"] = "1"

    print("실행 커맨드:")
    print(" ", " ".join(shlex.quote(c) for c in cmd))
    print(f"[info] DATA_DIR: {data_dir}")
    print(f"[info] TRAIN_LIST: {train_list}")
    print(f"[info] VAL_LIST: {val_list}")
    if selected_gpu:
        print(f"[info] 사용 GPU: {selected_gpu} (단일)")
    else:
        print("[info] GPU 미지정 → CPU 사용")

    # 실행: 하위 프로세스의 stdout/stderr를 그대로 콘솔에 출력하여 tqdm 표시 보존
    result = subprocess.run(cmd, cwd=str(repo_dir), env=env_vars)

    print(f"종료 코드: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
