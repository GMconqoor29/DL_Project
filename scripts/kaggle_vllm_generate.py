"""Script chạy trên Kaggle Notebook (GPU T4 x2 hoặc 1x T4) để sinh 2,650 mẫu bằng vLLM.

Cách chạy nhanh trên Kaggle (bật GPU T4 x2 + Internet ON):
    !pip install -q vllm
    !python kaggle_vllm_generate.py \
        --input /kaggle/input/<ten-dataset>/prompts_to_generate.jsonl \
        --output /kaggle/working/generated_continuations.jsonl

Đặc điểm thiết kế:
- Mặc định dùng **Base Models không khóa quyền (ungated)** trên Hugging Face:
    1) `Qwen/Qwen2.5-7B` (sinh mẫu Qwen In-domain)
    2) `unsloth/Meta-Llama-3.1-8B` (bản mở chính thức của Llama-3.1-8B Base, không cần xin quyền Meta/HF_TOKEN!)
  Vì dùng Base model (completion thuần túy), văn bản viết tiếp nối mạch trực tiếp vào câu thứ k của Người,
  KHÔNG BAO GIỜ xuất hiện câu dẫn nhập kiểu chatbot ("Dưới đây là phần viết tiếp...").
- Tự động phát hiện số lượng GPU T4:
    * Nếu có 2 x T4 (32GB VRAM): chạy `tensor_parallel_size=2, dtype="float16"` tốc độ tối đa (~1.5 giờ cho toàn bộ 2,650 mẫu).
    * Nếu chỉ có 1 x T4 (16GB VRAM): tự động bật lượng tử hóa 4-bit `bitsandbytes` để không bị tràn VRAM.
- Tự động lưu checkpoint sau mỗi `--save-every` (mặc định 200 mẫu): nếu phiên Kaggle bị ngắt, chạy lại lệnh cũ sẽ tự chạy tiếp phần còn thiếu!
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Set

# Ánh xạ tên generator chuẩn trong schema sang HuggingFace Repo ID mở (không cần HF_TOKEN)
DEFAULT_MODEL_MAP = {
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
}


def load_done_ids(output_path: Path) -> Set[str]:
    done: Set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    continue
    return done


def run_model_batch(
    hf_repo: str,
    items: List[Dict[str, Any]],
    output_path: Path,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> None:
    import torch
    from vllm import LLM, SamplingParams

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("Không tìm thấy GPU! Hãy bật Accelerator -> GPU T4 x2 trong Kaggle Settings.")

    llm_kwargs: Dict[str, Any] = {
        "model": hf_repo,
        "dtype": "float16",  # Bắt buộc float16 trên Tesla T4 (compute capability 7.5 không hỗ trợ bfloat16)
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": True,
    }
    if num_gpus >= 2:
        llm_kwargs["tensor_parallel_size"] = 2
        print(f"[vLLM] Khởi tạo {hf_repo} ở chế độ float16 trên 2 GPU (Tensor Parallel = 2)...", flush=True)
    else:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
        print(f"[vLLM] Chỉ có 1 GPU -> Khởi tạo {hf_repo} ở chế độ 4-bit bitsandbytes...", flush=True)

    llm = LLM(**llm_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompts = [x["prefix_text"] + " " for x in chunk]
        params_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.08,
                max_tokens=int(x.get("max_new_tokens", 380)),
            )
            for x in chunk
        ]

        outputs = llm.generate(prompts, params_list)
        with open(output_path, "a", encoding="utf-8") as out_f:
            for item, out in zip(chunk, outputs):
                gen_text = out.outputs[0].text if out.outputs else ""
                record = {
                    **item,
                    "hf_repo_used": hf_repo,
                    "raw_continuation": gen_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[{hf_repo}] Đã sinh và lưu {min(start + batch_size, len(items))}/{len(items)} mẫu.",
            flush=True,
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh phần viết tiếp văn bản bằng vLLM trên Kaggle T4.")
    parser.add_argument("--input", required=True, type=Path, help="Đường dẫn tới prompts_to_generate.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Đường dẫn lưu generated_continuations.jsonl")
    parser.add_argument("--batch-size", type=int, default=200, help="Số mẫu mỗi đợt sinh và lưu checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7, help="Nhiệt độ lấy mẫu (mặc định: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (mặc định: 0.9)")
    parser.add_argument("--qwen-repo", default=DEFAULT_MODEL_MAP["Qwen/Qwen2.5-7B"], help="HF repo cho Qwen")
    parser.add_argument("--llama-repo", default=DEFAULT_MODEL_MAP["meta-llama/Llama-3.1-8B"], help="HF repo cho Llama")
    args = parser.parse_args()

    model_map = {
        "Qwen/Qwen2.5-7B": args.qwen_repo,
        "meta-llama/Llama-3.1-8B": args.llama_repo,
    }

    done_ids = load_done_ids(args.output)
    print(f"Tìm thấy {len(done_ids)} mẫu đã hoàn thành từ lần chạy trước.", flush=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["sample_id"] in done_ids:
                continue
            grouped.setdefault(obj["generator"], []).append(obj)

    for gen_key, pending_items in grouped.items():
        hf_repo = model_map.get(gen_key, gen_key)
        print(f"=== Bắt đầu xử lý {len(pending_items)} mẫu cho generator {gen_key} ({hf_repo}) ===", flush=True)
        run_model_batch(
            hf_repo=hf_repo,
            items=pending_items,
            output_path=args.output,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    print("HOÀN TẤT SINH TOÀN BỘ DỮ LIỆU TRÊN KAGGLE!", flush=True)


if __name__ == "__main__":
    main()
