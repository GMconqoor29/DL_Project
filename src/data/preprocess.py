"""Pipeline Tiền xử lý Phần 1 (chạy ngay trên CPU Local):
1. Rút 3 lô bài gốc hoàn toàn rời nhau (SEED = 42) từ A1 (vietnamese-news), B8 (Wiki), B9 (Books):
   - Lô 1: 1,500 mẫu Human hoàn chỉnh (Train 1000 / Val 220 / Test 280)
   - Lô 2a: 1,200 mẫu AI thuần có sẵn (Sailor2-8B: Train 780, Val 150, Test 70 | Gemma-3-12B OOD Test: 200)
   - Lô 2b + Lô 3: Xuất 2,650 bài mồi sang `data/interim/prompts_to_generate.jsonl` để chạy vLLM (Qwen2.5-7B & Llama-3.1-8B)
2. Tạo xong 100% 3 bộ Stress-Test Mở Rộng vào `data/stress_test/`:
   - `test_vtp_domain.jsonl` (Văn học Vũ Trọng Phụng - A2)
   - `test_iccies_commercial.jsonl` (1,000 mẫu GPT-4o-mini / Gemini 1.5 / DeepSeek - A5, khử trùng với Train/Val)
   - `test_rewriting_robustness.jsonl` (4 cấp độ Paraphrase/Humanized - A4, lọc sạch lỗi tiếng Mã Lai)
3. Xuất `data/sample_debug.jsonl` (20 mẫu chuẩn) để đưa lên GitHub.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow.parquet as pq

# Cho phép chạy trực tiếp `python src/data/preprocess.py` từ thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.text_utils import (
    build_text_and_spans,
    clean_lines_and_edges,
    dedup_fingerprint,
    is_clean_vietnamese,
    split_sentences,
)

SEED = 42
EBOOK_NOISE_RE = re.compile(
    r"(?:ebook|tve-4u|quocsan|alpha\s*books|nhà\s+xuất\s+bản|nxb\s|bản\s+quyền|isbn|mục\s+lục|dịch\s+giả)",
    re.IGNORECASE,
)


def length_bin(num_words: int) -> str:
    if num_words < 220:
        return "short"
    if num_words < 380:
        return "medium"
    return "long"


def prepare_clean_doc(
    raw_text: str,
    trim_broken_leading: bool = False,
    min_sents: int = 6,
    max_sents: int = 28,
    min_words: int = 110,
    max_words: int = 620,
) -> Optional[Tuple[str, List[str], List[List[int]], int]]:
    """Làm sạch văn bản, tách câu và kiểm tra điều kiện độ dài."""
    cleaned = clean_lines_and_edges(
        raw_text,
        trim_broken_leading=trim_broken_leading,
        trim_broken_trailing=True,
    )
    if not is_clean_vietnamese(cleaned, min_words=min_words, max_words=max_words):
        return None

    sents = split_sentences(cleaned)
    if len(sents) < min_sents or len(sents) > max_sents:
        return None

    full_text, spans = build_text_and_spans(sents)
    n_words = len(full_text.split())
    if n_words < min_words or n_words > max_words:
        return None

    return full_text, sents, spans, n_words


def stratified_pick(
    candidates: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
    used_ids: Set[str],
    used_fps: Set[str],
) -> List[Dict[str, Any]]:
    """Lấy mẫu ngẫu nhiên phân tầng cân bằng theo phân bố độ dài (short / medium / long)."""
    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for item in candidates:
        if item["source_article_id"] in used_ids or item["fp"] in used_fps:
            continue
        buckets[length_bin(item["n_words"])].append(item)

    for b in buckets.values():
        rng.shuffle(b)

    # Tỷ lệ mục tiêu gần phân phối tự nhiên: 30% short, 45% medium, 25% long
    targets = {
        "short": int(round(k * 0.30)),
        "medium": int(round(k * 0.45)),
    }
    targets["long"] = k - targets["short"] - targets["medium"]

    picked: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for b_name in ("short", "medium", "long"):
        b_list = buckets[b_name]
        need = targets[b_name]
        for item in b_list:
            if item["source_article_id"] in used_ids or item["fp"] in used_fps:
                continue
            if need > 0:
                picked.append(item)
                used_ids.add(item["source_article_id"])
                used_fps.add(item["fp"])
                need -= 1
            else:
                leftovers.append(item)

    if len(picked) < k:
        rng.shuffle(leftovers)
        for item in leftovers:
            if len(picked) >= k:
                break
            if item["source_article_id"] in used_ids or item["fp"] in used_fps:
                continue
            picked.append(item)
            used_ids.add(item["source_article_id"])
            used_fps.add(item["fp"])

    if len(picked) < k:
        raise RuntimeError(f"Không đủ mẫu sạch để lấy {k} mẫu (chỉ tìm được {len(picked)}).")

    rng.shuffle(picked)
    return picked


def scan_news_parquet(
    parquet_path: Path,
    ai_col: str,
    generator_name: str,
    max_scan: int,
    rng: random.Random,
    row_slice: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Quét file Parquet của A1 (vietnamese-news) và thu thập bể ứng viên Human & AI sạch."""
    pf = pq.ParquetFile(parquet_path)
    table = pf.read(columns=["id", "text", ai_col])
    total = table.num_rows
    lo, hi = row_slice if row_slice is not None else (0, total)
    indices = list(range(lo, min(hi, total)))
    rng.shuffle(indices)

    ids = table["id"].to_pylist()
    human_texts = table["text"].to_pylist()
    ai_texts = table[ai_col].to_pylist()

    human_pool: List[Dict[str, Any]] = []
    ai_pool: List[Dict[str, Any]] = []

    for idx in indices:
        src_id = f"news_{ids[idx]}"
        if len(human_pool) < max_scan:
            res_h = prepare_clean_doc(human_texts[idx], trim_broken_leading=False)
            if res_h is not None:
                ft, sents, spans, nw = res_h
                human_pool.append(
                    {
                        "source_article_id": src_id,
                        "domain": "news",
                        "full_text": ft,
                        "sentences": sents,
                        "sentence_char_spans": spans,
                        "n_words": nw,
                        "fp": dedup_fingerprint(ft),
                    }
                )

        if len(ai_pool) < max_scan // 2:
            res_a = prepare_clean_doc(ai_texts[idx], trim_broken_leading=True)
            if res_a is not None:
                ft_a, sents_a, spans_a, nw_a = res_a
                ai_pool.append(
                    {
                        "source_article_id": src_id,
                        "domain": "news",
                        "generator": generator_name,
                        "full_text": ft_a,
                        "sentences": sents_a,
                        "sentence_char_spans": spans_a,
                        "n_words": nw_a,
                        "fp": dedup_fingerprint(ft_a),
                    }
                )

        if len(human_pool) >= max_scan and len(ai_pool) >= max_scan // 2:
            break

    return human_pool, ai_pool


def scan_wikipedia_pool(wiki_dir: Path, needed: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Quét các bài viết tiếng Việt sạch từ B8 (vietnamese-wikipedia-clean)."""
    part_files = sorted(wiki_dir.glob("viet_wiki_part_*.parquet"))
    pf = pq.ParquetFile(part_files[0])
    table = pf.read(columns=["page_id", "text"])
    indices = list(range(table.num_rows))
    rng.shuffle(indices)

    pids = table["page_id"].to_pylist()
    texts = table["text"].to_pylist()

    pool: List[Dict[str, Any]] = []
    for idx in indices:
        raw = texts[idx]
        if not raw or len(raw) < 600:
            continue
        # Lấy đoạn mở đầu ~2500 ký tự của trang Wiki để giữ mạch văn xuôi tự nhiên
        snippet = raw[:2600]
        res = prepare_clean_doc(snippet, trim_broken_leading=False, min_sents=6, max_sents=24)
        if res is None:
            continue
        ft, sents, spans, nw = res
        pool.append(
            {
                "source_article_id": f"wiki_{pids[idx]}",
                "domain": "wikipedia",
                "full_text": ft,
                "sentences": sents,
                "sentence_char_spans": spans,
                "n_words": nw,
                "fp": dedup_fingerprint(ft),
            }
        )
        if len(pool) >= needed:
            break
    return pool


def scan_books_pool(books_dir: Path, needed: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Trích đoạn giữa sách tiếng Việt sạch từ B9 (vietnamese-book-corpus), bỏ bìa/mục lục."""
    part_files = sorted((books_dir / "data").glob("train-*.parquet"))
    pf = pq.ParquetFile(part_files[0])
    table = pf.read(columns=["text"])
    indices = list(range(table.num_rows))
    rng.shuffle(indices)
    texts = table["text"].to_pylist()

    pool: List[Dict[str, Any]] = []
    for idx in indices:
        raw = texts[idx]
        if not raw or len(raw) < 12000:
            continue
        # Bỏ 25% đầu (bìa, lời tựa, mục lục) và 15% cuối (phụ lục), cắt một đoạn ~2400 ký tự ở giữa
        mid_start = int(len(raw) * rng.uniform(0.25, 0.70))
        chunk = raw[mid_start : mid_start + 2600]
        if EBOOK_NOISE_RE.search(chunk):
            continue
        res = prepare_clean_doc(chunk, trim_broken_leading=True, min_sents=6, max_sents=24)
        if res is None:
            continue
        ft, sents, spans, nw = res
        pool.append(
            {
                "source_article_id": f"book_s0_{idx:05d}",
                "domain": "books",
                "full_text": ft,
                "sentences": sents,
                "sentence_char_spans": spans,
                "n_words": nw,
                "fp": dedup_fingerprint(ft),
            }
        )
        if len(pool) >= needed:
            break
    return pool


def make_ready_record(
    item: Dict[str, Any],
    sample_id: str,
    split: str,
    doc_label: int,
    doc_type: str,
    generator: str,
    is_ood_generator: bool,
) -> Dict[str, Any]:
    sents = item["sentences"]
    n_sents = len(sents)
    return {
        "sample_id": sample_id,
        "source_article_id": item["source_article_id"],
        "split": split,
        "is_ood_generator": is_ood_generator,
        "domain": item["domain"],
        "doc_label": doc_label,
        "doc_type": doc_type,
        "generator": generator,
        "boundary_sentence_idx": -1 if doc_label == 0 else 0,
        "full_text": item["full_text"],
        "sentence_char_spans": item["sentence_char_spans"],
        "sentences": sents,
        "sentence_labels": [0] * n_sents if doc_label == 0 else [1] * n_sents,
    }


def make_generation_prompt(
    seed_item: Dict[str, Any],
    sample_id: str,
    split: str,
    target_doc_type: str,
    generator: str,
    is_ood_generator: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    """Tạo bản ghi mồi cho vLLM:
    - Nếu `target_doc_type == 'mixed_human_ai'`: cắt câu k thuộc [20%, 80%] giữ nguyên k câu đầu của Người.
    - Nếu `target_doc_type == 'ai'` (Qwen AI thuần): lấy 2 câu đầu làm ngữ cảnh mồi, sau khi sinh sẽ cắt bỏ 2 câu này.
    """
    sents = seed_item["sentences"]
    n = len(sents)
    if target_doc_type == "mixed_human_ai":
        lo = max(2, int(n * 0.20))
        hi = min(n - 2, max(lo, int(n * 0.80)))
        k = rng.randint(lo, hi)
        prefix_sents = sents[:k]
        tail_words = sum(len(s.split()) for s in sents[k:])
        target_words = max(80, min(420, tail_words))
    else:
        k = 2
        prefix_sents = sents[:2]
        target_words = max(160, min(420, seed_item["n_words"]))

    prefix_text = " ".join(prefix_sents)
    return {
        "sample_id": sample_id,
        "source_article_id": seed_item["source_article_id"],
        "split": split,
        "is_ood_generator": is_ood_generator,
        "domain": seed_item["domain"],
        "target_doc_type": target_doc_type,
        "generator": generator,
        "cut_sentence_idx": k,
        "drop_prefix_after_gen": target_doc_type == "ai",
        "prefix_sentences": prefix_sents,
        "prefix_text": prefix_text,
        "target_new_words": target_words,
        # 1 từ tiếng Việt ~ 1.45 subword tokens + cộng dư 80 tokens để không cụt câu sớm
        "max_new_tokens": min(700, max(160, int(target_words * 1.55) + 80)),
    }


def build_stress_tests(
    datasets_dir: Path,
    out_dir: Path,
    core_train_val_fps: Set[str],
    rng: random.Random,
) -> Dict[str, int]:
    """Xây dựng 3 bộ Stress-Test Mở Rộng hoàn chỉnh (Test-only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}

    # 1. Stress-Test 1: OOD Miền Văn Học Vũ Trọng Phụng (A2)
    s_pf = pq.ParquetFile(datasets_dir / "vu-trong-phung/Sailor2-8B-VuTrongPhung.parquet").read()
    g_pf = pq.ParquetFile(datasets_dir / "vu-trong-phung/Gemma-3-12B-VuTrongPhung.parquet").read()

    vtp_records: List[Dict[str, Any]] = []
    seen_vtp_human: Set[str] = set()

    for row in s_pf.to_pylist():
        src_id = f"vtp_{row['id'][:16]}"
        if src_id not in seen_vtp_human:
            res_h = prepare_clean_doc(row["text"], trim_broken_leading=False, min_sents=4, min_words=80)
            if res_h is not None:
                ft, sents, spans, nw = res_h
                seen_vtp_human.add(src_id)
                vtp_records.append(
                    make_ready_record(
                        {
                            "source_article_id": src_id,
                            "domain": "literature_vtp",
                            "full_text": ft,
                            "sentences": sents,
                            "sentence_char_spans": spans,
                        },
                        sample_id=f"{src_id}_human",
                        split="stress_test_vtp",
                        doc_label=0,
                        doc_type="human",
                        generator="human",
                        is_ood_generator=False,
                    )
                )
        res_s = prepare_clean_doc(
            row["sail-Sailor2-8B-Chat-hf_generated_text_wo_prompt"],
            trim_broken_leading=True,
            min_sents=4,
            min_words=80,
        )
        if res_s is not None:
            ft, sents, spans, nw = res_s
            vtp_records.append(
                make_ready_record(
                    {
                        "source_article_id": src_id,
                        "domain": "literature_vtp",
                        "full_text": ft,
                        "sentences": sents,
                        "sentence_char_spans": spans,
                    },
                    sample_id=f"{src_id}_sailor2",
                    split="stress_test_vtp",
                    doc_label=1,
                    doc_type="ai",
                    generator="sail/Sailor2-8B-Chat",
                    is_ood_generator=False,
                )
            )

    for row in g_pf.to_pylist():
        src_id = f"vtp_{row['id'][:16]}"
        res_g = prepare_clean_doc(
            row["google-gemma-3-12b-it-hf_generated_text_wo_prompt"],
            trim_broken_leading=True,
            min_sents=4,
            min_words=80,
        )
        if res_g is not None:
            ft, sents, spans, nw = res_g
            vtp_records.append(
                make_ready_record(
                    {
                        "source_article_id": src_id,
                        "domain": "literature_vtp",
                        "full_text": ft,
                        "sentences": sents,
                        "sentence_char_spans": spans,
                    },
                    sample_id=f"{src_id}_gemma3",
                    split="stress_test_vtp",
                    doc_label=1,
                    doc_type="ai",
                    generator="google/gemma-3-12b-it",
                    is_ood_generator=True,
                )
            )

    write_jsonl(out_dir / "test_vtp_domain.jsonl", vtp_records)
    stats["stress_vtp_domain"] = len(vtp_records)

    # 2. Stress-Test 2: Commercial LLMs (A5 - ICCIES 2025: 500 Human + 500 AI đã khử trùng với Train/Val)
    iccies_human: List[Dict[str, Any]] = []
    iccies_ai: List[Dict[str, Any]] = []
    seen_iccies_fp: Set[str] = set(core_train_val_fps)

    # Đọc ngẫu nhiên phân tán bằng Reservoir Sampling trên luồng 200k dòng để lấy đều cả GPT-4o, Gemini, DeepSeek
    human_candidates: List[Tuple[int, str]] = []
    ai_candidates: List[Tuple[int, str]] = []
    with open(datasets_dir / "iccies-2025-news-human-ai/Ai_human_200k.csv", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            lbl = row["Label"].strip()
            txt = row["Text"]
            if lbl == "0":
                if len(human_candidates) < 2500:
                    human_candidates.append((row_idx, txt))
                else:
                    j = rng.randint(0, row_idx)
                    if j < 2500:
                        human_candidates[j] = (row_idx, txt)
            elif lbl == "1":
                if len(ai_candidates) < 2500:
                    ai_candidates.append((row_idx, txt))
                else:
                    j = rng.randint(0, row_idx)
                    if j < 2500:
                        ai_candidates[j] = (row_idx, txt)

    rng.shuffle(human_candidates)
    rng.shuffle(ai_candidates)

    for row_idx, txt in human_candidates:
        if len(iccies_human) >= 500:
            break
        res = prepare_clean_doc(txt, trim_broken_leading=False, min_sents=5, min_words=100)
        if res is None:
            continue
        ft, sents, spans, _ = res
        fp = dedup_fingerprint(ft)
        if fp in seen_iccies_fp:
            continue
        seen_iccies_fp.add(fp)
        iccies_human.append(
            make_ready_record(
                {
                    "source_article_id": f"iccies_h_{row_idx}",
                    "domain": "news",
                    "full_text": ft,
                    "sentences": sents,
                    "sentence_char_spans": spans,
                },
                sample_id=f"iccies_{row_idx:06d}_human",
                split="stress_test_iccies",
                doc_label=0,
                doc_type="human",
                generator="human",
                is_ood_generator=False,
            )
        )

    for row_idx, txt in ai_candidates:
        if len(iccies_ai) >= 500:
            break
        res = prepare_clean_doc(txt, trim_broken_leading=False, min_sents=5, min_words=100)
        if res is None:
            continue
        ft, sents, spans, _ = res
        fp = dedup_fingerprint(ft)
        if fp in seen_iccies_fp:
            continue
        seen_iccies_fp.add(fp)
        iccies_ai.append(
            make_ready_record(
                {
                    "source_article_id": f"iccies_a_{row_idx}",
                    "domain": "news",
                    "full_text": ft,
                    "sentences": sents,
                    "sentence_char_spans": spans,
                },
                sample_id=f"iccies_{row_idx:06d}_commercial_ai",
                split="stress_test_iccies",
                doc_label=1,
                doc_type="ai",
                generator="commercial_llm_iccies",
                is_ood_generator=True,
            )
        )

    iccies_all = iccies_human + iccies_ai
    rng.shuffle(iccies_all)
    write_jsonl(out_dir / "test_iccies_commercial.jsonl", iccies_all)
    stats["stress_iccies_commercial"] = len(iccies_all)

    # 3. Stress-Test 3: Paraphrase & Humanizer Robustness (A4 - Text-Rewriting, lọc sạch tiếng Mã Lai)
    rewrite_records: List[Dict[str, Any]] = []
    seen_rw_human: Set[str] = set()
    variant_cols = [
        ("AI_text", "ai_original"),
        ("AI_paraphrase_1", "ai_paraphrase_1"),
        ("AI_paraphrase_2", "ai_paraphrase_2"),
        ("AI_humanized", "ai_humanized"),
    ]
    for p_str in sorted(glob.glob(str(datasets_dir / "text-rewriting/*.jsonl"))):
        p_name = Path(p_str).stem
        if "Sailor2-14B" in p_name:
            gen = "sail/Sailor2-14B-Chat"
        elif "Sailor2-8B" in p_name:
            gen = "sail/Sailor2-8B-Chat"
        else:
            gen = "google/gemma-3-12b-it"

        with open(p_str, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                obj = json.loads(line)
                fp_h = dedup_fingerprint(obj["text"], 40)
                src_id = f"vtp_rw_{line_idx:03d}"

                if fp_h not in seen_rw_human:
                    res_h = prepare_clean_doc(obj["text"], trim_broken_leading=False, min_sents=4, min_words=80)
                    if res_h is not None:
                        seen_rw_human.add(fp_h)
                        ft, sents, spans, _ = res_h
                        rec = make_ready_record(
                            {
                                "source_article_id": src_id,
                                "domain": "literature_vtp",
                                "full_text": ft,
                                "sentences": sents,
                                "sentence_char_spans": spans,
                            },
                            sample_id=f"{src_id}_human",
                            split="stress_test_rewriting",
                            doc_label=0,
                            doc_type="human",
                            generator="human",
                            is_ood_generator=False,
                        )
                        rec["rewrite_stage"] = "human"
                        rewrite_records.append(rec)

                for col_key, stage_name in variant_cols:
                    raw_val = obj.get(col_key, "")
                    res_v = prepare_clean_doc(raw_val, trim_broken_leading=True, min_sents=4, min_words=70)
                    if res_v is None:
                        continue
                    ft, sents, spans, _ = res_v
                    short_gen = gen.split("/")[-1].lower().replace("-", "_")
                    rec = make_ready_record(
                        {
                            "source_article_id": src_id,
                            "domain": "literature_vtp",
                            "full_text": ft,
                            "sentences": sents,
                            "sentence_char_spans": spans,
                        },
                        sample_id=f"{src_id}_{short_gen}_{stage_name}",
                        split="stress_test_rewriting",
                        doc_label=1,
                        doc_type="ai",
                        generator=gen,
                        is_ood_generator=True,
                    )
                    rec["rewrite_stage"] = stage_name
                    rewrite_records.append(rec)

    write_jsonl(out_dir / "test_rewriting_robustness.jsonl", rewrite_records)
    stats["stress_rewriting_robustness"] = len(rewrite_records)
    return stats


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    datasets_dir = PROJECT_ROOT / "datasets"
    interim_dir = PROJECT_ROOT / "data" / "interim"
    stress_dir = PROJECT_ROOT / "data" / "stress_test"

    print("[1/5] Đang quét và làm sạch bể ứng viên từ A1 (vietnamese-news), B8 (Wiki), B9 (Books)...", flush=True)
    # A1 Train
    train_news_h, train_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/train.parquet",
        ai_col="sail-Sailor2-8B-Chat-generated_text",
        generator_name="sail/Sailor2-8B-Chat",
        max_scan=7500,
        rng=rng,
    )
    # A1 Validation chia đôi theo dòng không chồng lấn: nửa đầu cho VAL, nửa sau cho TEST In-Domain
    val_pf_rows = pq.ParquetFile(datasets_dir / "vietnamese-news/validation.parquet").metadata.num_rows
    mid_val = int(val_pf_rows * 0.60)
    val_news_h, val_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/validation.parquet",
        ai_col="sail-Sailor2-8B-Chat-generated_text",
        generator_name="sail/Sailor2-8B-Chat",
        max_scan=2500,
        rng=rng,
        row_slice=(0, mid_val),
    )
    test_id_news_h, test_id_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/validation.parquet",
        ai_col="sail-Sailor2-8B-Chat-generated_text",
        generator_name="sail/Sailor2-8B-Chat",
        max_scan=1800,
        rng=rng,
        row_slice=(mid_val, val_pf_rows),
    )
    # A1 Test (Gemma-3-12B OOD)
    test_ood_news_h, test_ood_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/test.parquet",
        ai_col="google-gemma-3-12b-it-generated_text",
        generator_name="google/gemma-3-12b-it",
        max_scan=2800,
        rng=rng,
    )

    # B8 Wikipedia & B9 Books
    wiki_pool = scan_wikipedia_pool(datasets_dir / "vietnamese-wikipedia-clean", needed=900, rng=rng)
    books_pool = scan_books_pool(datasets_dir / "vietnamese-book-corpus", needed=600, rng=rng)

    # Chia bể Wiki & Books hoàn toàn độc lập cho Train / Val / Test
    wiki_train, wiki_val, wiki_test = wiki_pool[:550], wiki_pool[550:700], wiki_pool[700:]
    books_train, books_val, books_test = books_pool[:350], books_pool[350:460], books_pool[460:]

    print(
        f"   -> Bể sạch thu được: TrainNewsH={len(train_news_h)}, TrainNewsAI={len(train_news_ai)}, "
        f"ValNewsH={len(val_news_h)}, TestIDNewsH={len(test_id_news_h)}, TestOODNewsH={len(test_ood_news_h)}, "
        f"Wiki={len(wiki_pool)}, Books={len(books_pool)}",
        flush=True,
    )

    print("[2/5] Đang rút 3 Lô bài gốc rời nhau (Stratified + SEED=42)...", flush=True)
    used_ids: Set[str] = set()
    used_fps: Set[str] = set()

    ready_records: List[Dict[str, Any]] = []
    prompt_records: List[Dict[str, Any]] = []

    # ================= TRAIN (3,500 mẫu: 1,000 Human + 1,000 AI [780 Sailor2 + 220 Qwen] + 1,500 Mixed Qwen) =================
    tr_h_news = stratified_pick(train_news_h, 850, rng, used_ids, used_fps)
    tr_h_wiki = stratified_pick(wiki_train, 100, rng, used_ids, used_fps)
    tr_h_books = stratified_pick(books_train, 50, rng, used_ids, used_fps)
    for i, item in enumerate(tr_h_news + tr_h_wiki + tr_h_books):
        ready_records.append(
            make_ready_record(item, f"train_human_{i:04d}", "train", 0, "human", "human", False)
        )

    tr_ai_sailor = stratified_pick(train_news_ai, 780, rng, used_ids, used_fps)
    for i, item in enumerate(tr_ai_sailor):
        ready_records.append(
            make_ready_record(item, f"train_ai_sailor2_{i:04d}", "train", 1, "ai", item["generator"], False)
        )

    # 220 mồi cho Qwen AI thuần (Train)
    tr_seed_qwen_ai = stratified_pick(train_news_h, 220, rng, used_ids, used_fps)
    for i, item in enumerate(tr_seed_qwen_ai):
        prompt_records.append(
            make_generation_prompt(item, f"train_ai_qwen_{i:04d}", "train", "ai", "Qwen/Qwen2.5-7B", False, rng)
        )

    # 1,500 mồi cho Qwen Mixed (Train): 1,250 News + 150 Wiki + 100 Books
    tr_mix_seeds = (
        stratified_pick(train_news_h, 1250, rng, used_ids, used_fps)
        + stratified_pick(wiki_train, 150, rng, used_ids, used_fps)
        + stratified_pick(books_train, 100, rng, used_ids, used_fps)
    )
    rng.shuffle(tr_mix_seeds)
    for i, item in enumerate(tr_mix_seeds):
        prompt_records.append(
            make_generation_prompt(
                item, f"train_mixed_qwen_{i:04d}", "train", "mixed_human_ai", "Qwen/Qwen2.5-7B", False, rng
            )
        )

    # ================= VAL (750 mẫu: 220 Human + 200 AI [150 Sailor2 + 50 Qwen] + 330 Mixed Qwen) =================
    va_h = (
        stratified_pick(val_news_h, 185, rng, used_ids, used_fps)
        + stratified_pick(wiki_val, 20, rng, used_ids, used_fps)
        + stratified_pick(books_val, 15, rng, used_ids, used_fps)
    )
    for i, item in enumerate(va_h):
        ready_records.append(make_ready_record(item, f"val_human_{i:04d}", "val", 0, "human", "human", False))

    va_ai_sailor = stratified_pick(val_news_ai, 150, rng, used_ids, used_fps)
    for i, item in enumerate(va_ai_sailor):
        ready_records.append(
            make_ready_record(item, f"val_ai_sailor2_{i:04d}", "val", 1, "ai", item["generator"], False)
        )

    va_seed_qwen_ai = stratified_pick(val_news_h, 50, rng, used_ids, used_fps)
    for i, item in enumerate(va_seed_qwen_ai):
        prompt_records.append(
            make_generation_prompt(item, f"val_ai_qwen_{i:04d}", "val", "ai", "Qwen/Qwen2.5-7B", False, rng)
        )

    va_mix_seeds = (
        stratified_pick(val_news_h, 280, rng, used_ids, used_fps)
        + stratified_pick(wiki_val, 30, rng, used_ids, used_fps)
        + stratified_pick(books_val, 20, rng, used_ids, used_fps)
    )
    rng.shuffle(va_mix_seeds)
    for i, item in enumerate(va_mix_seeds):
        prompt_records.append(
            make_generation_prompt(
                item, f"val_mixed_qwen_{i:04d}", "val", "mixed_human_ai", "Qwen/Qwen2.5-7B", False, rng
            )
        )

    # Lưu tập fingerprint của Train + Val để khử trùng cho Stress-Test
    train_val_fps = set(used_fps)

    # ================= CORE TEST (1,100 mẫu = 500 In-Domain + 600 OOD Generator) =================
    # Human Test: 280 mẫu (120 từ val_tail + 110 từ test.parquet + 30 Wiki + 20 Books)
    te_h = (
        stratified_pick(test_id_news_h, 120, rng, used_ids, used_fps)
        + stratified_pick(test_ood_news_h, 110, rng, used_ids, used_fps)
        + stratified_pick(wiki_test, 30, rng, used_ids, used_fps)
        + stratified_pick(books_test, 20, rng, used_ids, used_fps)
    )
    for i, item in enumerate(te_h):
        ready_records.append(make_ready_record(item, f"test_human_{i:04d}", "test", 0, "human", "human", False))

    # AI thuần Test (300 mẫu): 70 Sailor2 (ID) + 30 Qwen (ID) + 200 Gemma-3-12B (OOD)
    te_ai_sailor = stratified_pick(test_id_news_ai, 70, rng, used_ids, used_fps)
    for i, item in enumerate(te_ai_sailor):
        ready_records.append(
            make_ready_record(item, f"test_ai_sailor2_{i:04d}", "test", 1, "ai", item["generator"], False)
        )

    te_seed_qwen_ai = stratified_pick(test_id_news_h, 30, rng, used_ids, used_fps)
    for i, item in enumerate(te_seed_qwen_ai):
        prompt_records.append(
            make_generation_prompt(item, f"test_ai_qwen_{i:04d}", "test", "ai", "Qwen/Qwen2.5-7B", False, rng)
        )

    te_ai_gemma = stratified_pick(test_ood_news_ai, 200, rng, used_ids, used_fps)
    for i, item in enumerate(te_ai_gemma):
        ready_records.append(
            make_ready_record(item, f"test_ai_gemma3_{i:04d}", "test", 1, "ai", item["generator"], True)
        )

    # Mixed Test (520 mẫu): 120 Qwen (ID) + 400 Llama-3.1-8B (OOD)
    te_mix_qwen_seeds = (
        stratified_pick(test_id_news_h, 95, rng, used_ids, used_fps)
        + stratified_pick(wiki_test, 15, rng, used_ids, used_fps)
        + stratified_pick(books_test, 10, rng, used_ids, used_fps)
    )
    rng.shuffle(te_mix_qwen_seeds)
    for i, item in enumerate(te_mix_qwen_seeds):
        prompt_records.append(
            make_generation_prompt(
                item, f"test_mixed_qwen_{i:04d}", "test", "mixed_human_ai", "Qwen/Qwen2.5-7B", False, rng
            )
        )

    te_mix_llama_seeds = (
        stratified_pick(test_ood_news_h, 330, rng, used_ids, used_fps)
        + stratified_pick(wiki_test, 45, rng, used_ids, used_fps)
        + stratified_pick(books_test, 25, rng, used_ids, used_fps)
    )
    rng.shuffle(te_mix_llama_seeds)
    for i, item in enumerate(te_mix_llama_seeds):
        prompt_records.append(
            make_generation_prompt(
                item,
                f"test_mixed_llama_{i:04d}",
                "test",
                "mixed_human_ai",
                "meta-llama/Llama-3.1-8B",
                True,
                rng,
            )
        )

    print("[3/5] Đang ghi dữ liệu Phần 1 (`core_ready_part1.jsonl` & `prompts_to_generate.jsonl`)...", flush=True)
    write_jsonl(interim_dir / "core_ready_part1.jsonl", ready_records)
    write_jsonl(interim_dir / "prompts_to_generate.jsonl", prompt_records)

    # Tạo `data/sample_debug.jsonl` (20 mẫu cân bằng, có thêm mẫu mô phỏng Mixed để test code Boundary ngay)
    debug_samples = (
        [r for r in ready_records if r["split"] == "train" and r["doc_type"] == "human"][:6]
        + [r for r in ready_records if r["split"] == "train" and r["doc_type"] == "ai"][:5]
        + [r for r in ready_records if r["split"] == "test" and r["is_ood_generator"]][:4]
    )
    # Tạo thêm 5 mẫu Mixed giả lập từ cặp bài báo thật trong A1 (ghép k câu đầu Human + câu đuôi Sailor2) chỉ riêng cho file debug 20 dòng!
    for d_idx in range(5):
        h_rec = ready_records[d_idx]
        a_rec = [r for r in ready_records if r["doc_type"] == "ai"][d_idx]
        k_cut = max(2, len(h_rec["sentences"]) // 2)
        mix_sents = h_rec["sentences"][:k_cut] + a_rec["sentences"][: max(3, len(a_rec["sentences"]) // 2)]
        ft_m, sp_m = build_text_and_spans(mix_sents)
        debug_samples.append(
            {
                "sample_id": f"debug_mixed_mock_{d_idx:02d}",
                "source_article_id": f"debug_src_{d_idx:02d}",
                "split": "train" if d_idx < 3 else "test",
                "is_ood_generator": False,
                "domain": "news",
                "doc_label": 1,
                "doc_type": "mixed_human_ai",
                "generator": "sail/Sailor2-8B-Chat",
                "boundary_sentence_idx": k_cut,
                "full_text": ft_m,
                "sentence_char_spans": sp_m,
                "sentences": mix_sents,
                "sentence_labels": [0] * k_cut + [1] * (len(mix_sents) - k_cut),
            }
        )
    write_jsonl(PROJECT_ROOT / "data" / "sample_debug.jsonl", debug_samples)

    print("[4/5] Đang tạo 3 bộ Stress-Test Mở Rộng (`data/stress_test/`)...", flush=True)
    stress_stats = build_stress_tests(datasets_dir, stress_dir, train_val_fps, rng)

    print("[5/5] Đang kiểm tra tính toàn vẹn (Zero Leakage & Thống kê phân phối)...", flush=True)
    all_src_by_split: Dict[str, Set[str]] = collections.defaultdict(set)
    for r in ready_records:
        all_src_by_split[r["split"]].add(r["source_article_id"])
    for p in prompt_records:
        all_src_by_split[p["split"]].add(p["source_article_id"])

    assert not (all_src_by_split["train"] & all_src_by_split["val"]), "Rò rỉ source_id giữa Train và Val!"
    assert not (all_src_by_split["train"] & all_src_by_split["test"]), "Rò rỉ source_id giữa Train và Test!"
    assert not (all_src_by_split["val"] & all_src_by_split["test"]), "Rò rỉ source_id giữa Val và Test!"

    summary = {
        "core_ready_count": len(ready_records),
        "prompts_to_generate_count": len(prompt_records),
        "projected_core_total": len(ready_records) + len(prompt_records),
        "ready_by_split_and_type": dict(
            collections.Counter(f"{r['split']}:{r['doc_type']}:{r['generator']}" for r in ready_records)
        ),
        "prompts_by_split_and_type": dict(
            collections.Counter(f"{p['split']}:{p['target_doc_type']}:{p['generator']}" for p in prompt_records)
        ),
        "avg_words_ready_human": round(
            sum(len(r["full_text"].split()) for r in ready_records if r["doc_label"] == 0)
            / max(1, sum(1 for r in ready_records if r["doc_label"] == 0)),
            1,
        ),
        "avg_words_ready_ai": round(
            sum(len(r["full_text"].split()) for r in ready_records if r["doc_label"] == 1)
            / max(1, sum(1 for r in ready_records if r["doc_label"] == 1)),
            1,
        ),
        "stress_tests": stress_stats,
        "sample_debug_count": len(debug_samples),
    }
    with open(interim_dir / "part1_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("HOÀN TẤT PHẦN 1! Thống kê tóm tắt:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
