"""Script dùng chung cho các thành viên tải bộ dữ liệu chuẩn từ Hugging Face Hub.

Cách dùng:
    1. Copy `.env.example` thành `.env` (điền `HF_TOKEN` nếu repo là Private).
    2. Chạy:
           python scripts/download_data.py
       Hoặc chỉ tải một thư mục con (ví dụ chỉ lấy `processed` hoặc `features_cache`):
           python scripts/download_data.py --subdir processed
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_file(env_path: Path) -> Dict[str, str]:
    """Đọc file `.env` đơn giản bằng Python thuần (không cần cài thêm python-dotenv)."""
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            loaded[key] = val
    return loaded


def main() -> None:
    # Ưu tiên biến môi trường hệ thống -> `.env` -> `.env.example`
    env_defaults = load_dotenv_file(PROJECT_ROOT / ".env.example")
    env_local = load_dotenv_file(PROJECT_ROOT / ".env")
    merged_env = {**env_defaults, **env_local}

    default_repo = os.environ.get("HF_DATASET_REPO") or merged_env.get("HF_DATASET_REPO") or "DLProject05/vi-mixed-ai-detection"
    default_token = os.environ.get("HF_TOKEN") or merged_env.get("HF_TOKEN") or None

    parser = argparse.ArgumentParser(description="Tải bộ dữ liệu chuẩn của nhóm từ Hugging Face Hub.")
    parser.add_argument("--repo-id", default=default_repo, help=f"HF Dataset Repo (mặc định: {default_repo})")
    parser.add_argument("--local-dir", type=Path, default=PROJECT_ROOT / "data", help="Thư mục lưu dữ liệu (mặc định: data/)")
    parser.add_argument(
        "--subdir",
        nargs="*",
        default=None,
        help="Chỉ tải các thư mục con cụ thể, ví dụ: --subdir processed stress_test features_cache",
    )
    parser.add_argument("--token", default=default_token, help="HF Access Token (bỏ trống nếu repo Public hoặc đã hf auth login)")
    args = parser.parse_args()

    allow_patterns = [f"{s.strip('/')}/*" for s in args.subdir] if args.subdir else None

    print(f"Đang tải dataset `{args.repo_id}` -> `{args.local_dir}`...", flush=True)
    if allow_patterns:
        print(f"  Chỉ lấy các mục: {allow_patterns}", flush=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.local_dir),
        allow_patterns=allow_patterns,
        token=args.token or None,
    )
    print("Tải dữ liệu hoàn tất!", flush=True)


if __name__ == "__main__":
    main()
