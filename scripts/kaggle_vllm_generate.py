"""Script chạy trên Kaggle Notebook (GPU T4 x2) để sinh dữ liệu tiếp nối bằng vLLM theo v3.

Cách chạy nhanh trên Kaggle (bật Accelerator -> GPU T4 x2 + Internet ON):
    !pip install -q -U vllm bitsandbytes
    !python scripts/kaggle_vllm_generate.py \
        --input /kaggle/input/<ten-dataset>/prompts_delta_to_generate.jsonl \
        --output /kaggle/working/generated_continuations.jsonl

Đặc điểm thiết kế v3+:
- Dùng Base Models mở (ungated): `Qwen/Qwen2.5-7B` (ID) & `unsloth/Meta-Llama-3.1-8B` (OOD).
- Prompt **không có dấu cách thừa ở cuối** (`prefix_text.rstrip()`) chống artifact BPE `"Ở "` tại ranh giới.
- Chống dừng sớm (`raw_continuation == ""`) triệt để cho `Qwen2.5-7B`:
    1) Bật `ignore_eos=True` + `logit_bias={tid: -100.0 for tid in tok.all_special_ids}` chặn toàn bộ
       special token (`<|endoftext|>`, `<|im_end|>`, `<|im_start|>`, ...).
    2) Vòng lặp **retry ngay trên GPU (tối đa 4 lần)** cho riêng những mẫu có đoạn văn đầu trước `\n\n`
       quá ngắn hoặc chứa từ khóa rác.
- Mặc định bắt buộc **2 x T4 (fp16, tensor_parallel_size=2)**; ghi đầy đủ `gen_config`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DEFAULT_MODEL_MAP = {
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
}

FORBIDDEN_GEN_RE = re.compile(
    r"(?:Dưới đây là|Phần tiếp theo|Tin liên quan|Xem thêm|Tags|Bài viết liên quan)",
    re.IGNORECASE,
)


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


def _compute_min_tokens(item: Dict[str, Any], max_tok: int) -> int:
    if item.get("drop_prefix_after_gen", False):
        raw_min = min(260, max(180, int(item.get("target_new_words", 220) * 1.0)))
    else:
        raw_min = min(100, max(55, int(item.get("target_new_words", 100) * 0.8)))
    return max(16, min(max_tok - 16, raw_min))


def _is_acceptable_raw(gen_text: str, is_pure_ai: bool) -> bool:
    if not gen_text:
        return False
    txt = gen_text.replace("\r\n", "\n").replace("\r", "\n").lstrip()
    if not txt or FORBIDDEN_GEN_RE.search(txt):
        return False
    dbl = txt.find("\n\n")
    first_block = txt[:dbl] if dbl != -1 else txt
    min_words = 175 if is_pure_ai else 45
    return len(first_block.split()) >= min_words


def _make_sampling_params(
    SamplingParams: Any,
    item: Dict[str, Any],
    temperature: float,
    top_p: float,
    special_bias: Optional[Dict[int, float]],
    seed: Optional[int] = None,
) -> Any:
    max_tok = int(item.get("max_new_tokens", 380))
    min_tok = _compute_min_tokens(item, max_tok)
    base_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": 1.08,
        "ignore_eos": True,
        "min_tokens": min_tok,
        "max_tokens": max_tok,
    }
    if seed is not None:
        base_kwargs["seed"] = seed
    if special_bias:
        try:
            return SamplingParams(**base_kwargs, logit_bias=special_bias)
        except Exception:
            pass
    return SamplingParams(**base_kwargs)


def run_model_batch(
    hf_repo: str,
    items: List[Dict[str, Any]],
    output_path: Path,
    batch_size: int,
    temperature: float,
    top_p: float,
    allow_4bit: bool = False,
    max_retries: int = 4,
) -> None:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"[Env] torch={torch.__version__}, vllm={vllm.__version__}", flush=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("Không tìm thấy GPU! Hãy bật Accelerator -> GPU T4 x2 trong Kaggle Settings.")

    llm_kwargs: Dict[str, Any] = {
        "model": hf_repo,
        "dtype": "float16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": True,
    }
    if num_gpus >= 2:
        llm_kwargs["tensor_parallel_size"] = 2
        dtype_used = "fp16"
        print(f"[vLLM] Khởi tạo {hf_repo} ở chế độ fp16 trên 2 GPU (TP=2)...", flush=True)
    elif allow_4bit:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
        dtype_used = "4bit_bnb"
        print(f"[vLLM] Chỉ có 1 GPU (--allow-4bit) -> Chạy {hf_repo} ở 4-bit bitsandbytes...", flush=True)
    else:
        raise RuntimeError(
            "Chỉ phát hiện 1 GPU! Kế hoạch v3 cấm tự rơi xuống 4-bit ngầm. "
            "Hãy bật GPU T4 x2 trên Kaggle hoặc truyền cờ --allow-4bit để ghi nhận rõ cấu hình."
        )

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    special_ids = getattr(tok, "all_special_ids", []) or []
    special_bias: Optional[Dict[int, float]] = {int(tid): -100.0 for tid in special_ids} if special_ids else None
    print(f"[vLLM] Chặn {len(special_ids)} special tokens bằng logit_bias + ignore_eos=True.", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompts = [x["prefix_text"].rstrip() for x in chunk]
        params_list = [_make_sampling_params(SamplingParams, x, temperature, top_p, special_bias, seed=42) for x in chunk]

        outputs = llm.generate(prompts, params_list)
        best_texts = [out.outputs[0].text if out.outputs else "" for out in outputs]

        # Retry ngay trên GPU cho riêng những mẫu bị rỗng / cắt sớm trước \n\n
        for attempt in range(1, max_retries + 1):
            bad_indices = [
                idx
                for idx, (it, txt) in enumerate(zip(chunk, best_texts))
                if not _is_acceptable_raw(txt, bool(it.get("drop_prefix_after_gen", False)))
            ]
            if not bad_indices:
                break
            retry_temp = min(0.95, round(temperature + 0.05 * attempt, 2))
            print(
                f"[{hf_repo}] Retry lần {attempt}/{max_retries} cho {len(bad_indices)}/{len(chunk)} mẫu ngắn/rỗng (temp={retry_temp})...",
                flush=True,
            )
            sub_prompts = [prompts[i] for i in bad_indices]
            sub_params = [
                _make_sampling_params(SamplingParams, chunk[i], retry_temp, top_p, special_bias, seed=42 + attempt * 97 + i)
                for i in bad_indices
            ]
            try:
                sub_outs = llm.generate(sub_prompts, sub_params)
            except Exception:
                # Fallback nếu vLLM build từ chối logit_bias trong runtime
                special_bias = None
                sub_params = [
                    _make_sampling_params(SamplingParams, chunk[i], retry_temp, top_p, None, seed=42 + attempt * 97 + i)
                    for i in bad_indices
                ]
                sub_outs = llm.generate(sub_prompts, sub_params)

            for i, out in zip(bad_indices, sub_outs):
                cand = out.outputs[0].text if out.outputs else ""
                if len(cand.split()) > len(best_texts[i].split()):
                    best_texts[i] = cand

        with open(output_path, "a", encoding="utf-8") as out_f:
            for item, gen_text in zip(chunk, best_texts):
                record = {
                    **item,
                    "hf_repo_used": hf_repo,
                    "gen_config": {
                        "repo": hf_repo,
                        "vllm_version": vllm.__version__,
                        "dtype": dtype_used,
                        "temperature": temperature,
                        "top_p": top_p,
                        "repetition_penalty": 1.08,
                        "ignore_eos": True,
                    },
                    "raw_continuation": gen_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        ok_cnt = sum(
            1 for it, txt in zip(chunk, best_texts) if _is_acceptable_raw(txt, bool(it.get("drop_prefix_after_gen", False)))
        )
        print(
            f"[{hf_repo}] Đã lưu {min(start + batch_size, len(items))}/{len(items)} mẫu (đạt độ dài ngay trên GPU: {ok_cnt}/{len(chunk)}).",
            flush=True,
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh phần viết tiếp văn bản bằng vLLM trên Kaggle T4 x2 (v3+).")
    parser.add_argument("--input", required=True, type=Path, help="Đường dẫn tới prompts_delta_to_generate.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Đường dẫn lưu generated_continuations.jsonl")
    parser.add_argument("--batch-size", type=int, default=200, help="Số mẫu mỗi đợt sinh và lưu checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7, help="Nhiệt độ lấy mẫu (mặc định: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (mặc định: 0.9)")
    parser.add_argument("--qwen-repo", default=DEFAULT_MODEL_MAP["Qwen/Qwen2.5-7B"], help="HF repo cho Qwen")
    parser.add_argument("--llama-repo", default=DEFAULT_MODEL_MAP["meta-llama/Llama-3.1-8B"], help="HF repo cho Llama")
    parser.add_argument("--allow-4bit", action="store_true", help="Cho phép rơi xuống 4-bit nếu chỉ có 1 GPU")
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
            line = line.strip()
            if not line:
                continue
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
            allow_4bit=args.allow_4bit,
        )

    print("HOÀN TẤT SINH TOÀN BỘ DỮ LIỆU TRÊN KAGGLE (v3+)!", flush=True)


if __name__ == "__main__":
    main()
