"""Pipeline Tiền xử lý Phần 1 (chạy trên CPU Local) theo Detailed_Project_Plan_v3.md:
1. Làm sạch & cắt đầu đối xứng (Symmetric Leading Cut) cho cả Human và AI từ A1, B8 (Wiki), B9 (Books),
   chuẩn hóa dấu câu đồng nhất và áp dụng một hàm `length_ok()` chung (6-24 câu, 150-650 từ).
2. Rút các lô bài gốc hoàn toàn rời nhau (SEED = 42, MinHash 5-gram Jaccard < 0.8):
   - Core Ready (2,740 mẫu): Human 1,540 + AI Sailor2 1,000 + AI Gemma-3 200.
   - Prompts (+20% dư = 3,300 mồi cho mục tiêu 2,750 mẫu vLLM): Qwen2.5-7B (ID) & Llama-3.1-8B (OOD).
   - Human Calibration Pool (1,000 mẫu): `data/calib/val_calib.jsonl` (500) & `test_calib.jsonl` (500).
3. Dựng 3 bộ Stress-Test Mở Rộng vào `data/stress_test/` (cùng quy tắc cắt đầu đối xứng & MinHash dedup).
4. Xuất `data/sample_debug.jsonl` (20 mẫu chuẩn, không dùng dữ liệu test thật cho việc debug).
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.text_utils import (
    MinHashLSHIndex,
    build_text_and_spans,
    clean_lines_and_edges,
    dedup_fingerprint,
    is_clean_vietnamese,
    length_ok,
    split_sentences,
)

SEED = 42
EBOOK_NOISE_RE = re.compile(
    r"(?:ebook|tve-4u|quocsan|alpha\s*books|nhà\s+xuất\s+bản|nxb\s|bản\s+quyền|isbn|mục\s+lục|dịch\s+giả)",
    re.IGNORECASE,
)


def length_bin(num_words: int) -> str:
    if num_words < 240:
        return "short"
    if num_words < 400:
        return "medium"
    return "long"


def prepare_clean_doc(
    raw_text: str,
    trim_broken_leading: bool = False,
    min_sents: int = 6,
    max_sents: int = 24,
    min_words: int = 150,
    max_words: int = 650,
) -> Optional[Tuple[str, List[str], List[List[int]], int]]:
    """Làm sạch văn bản, tách câu và kiểm tra điều kiện độ dài chung."""
    cleaned = clean_lines_and_edges(
        raw_text,
        trim_broken_leading=trim_broken_leading,
        trim_broken_trailing=True,
    )
    if not is_clean_vietnamese(cleaned, min_words=min_words, max_words=max_words):
        return None

    sents = split_sentences(cleaned)
    if trim_broken_leading and len(sents) >= 8:
        # Cắt đối xứng 1 câu chuyển ý mở đầu và 1 câu tóm tắt kết bài
        sents = sents[1:-1]
    full_text, spans = build_text_and_spans(sents)
    if not length_ok(
        sents,
        full_text,
        min_sents=min_sents,
        max_sents=max_sents,
        min_words=min_words,
        max_words=max_words,
    ):
        return None

    n_words = len(full_text.split())
    return full_text, sents, spans, n_words


def _strip_prompt_prefix(raw_text: str, user_prompt: str, fallback_chars: int = 160) -> str:
    """Cắt bỏ phần mở đầu ~50 token để mọi bài A1 (Human & AI) bắt đầu từ giữa bài."""
    if not raw_text:
        return ""
    up = (user_prompt or "").strip()
    stripped = raw_text.lstrip()
    if up and stripped.startswith(up):
        return stripped[len(up) :]
    if up and len(up) >= 30 and stripped.startswith(up[:30]):
        return stripped[len(up) :]
    return stripped[fallback_chars:] if len(stripped) > fallback_chars else ""


def stratified_pick(
    candidates: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
    used_ids: Set[str],
    used_fps: Set[str],
    lsh_index: Optional[MinHashLSHIndex] = None,
    preferred_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Lấy mẫu phân tầng theo độ dài, chống trùng lặp source_article_id, fingerprint và MinHash LSH."""
    if k == 0:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for item in candidates:
        if item["source_article_id"] in used_ids or item["fp"] in used_fps:
            continue
        buckets[length_bin(item["n_words"])].append(item)

    for b_name, b_list in buckets.items():
        rng.shuffle(b_list)
        if preferred_ids:
            # Ưu tiên các ID đã có trong lần sinh trước (nếu thuộc đúng bucket)
            b_list.sort(key=lambda x: 0 if x["source_article_id"] in preferred_ids else 1)

    targets = {
        "short": int(round(k * 0.30)),
        "medium": int(round(k * 0.45)),
    }
    targets["long"] = k - targets["short"] - targets["medium"]

    picked: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    def _accept(item: Dict[str, Any]) -> bool:
        sid = item["source_article_id"]
        fp = item["fp"]
        if sid in used_ids or fp in used_fps:
            return False
        if lsh_index is not None and lsh_index.query_duplicates(item["full_text"]):
            return False
        used_ids.add(sid)
        used_fps.add(fp)
        if lsh_index is not None:
            lsh_index.add(sid, item["full_text"])
        return True

    for b_name in ("short", "medium", "long"):
        b_list = buckets[b_name]
        need = targets[b_name]
        for item in b_list:
            if need > 0:
                if _accept(item):
                    picked.append(item)
                    need -= 1
            else:
                leftovers.append(item)

    if len(picked) < k:
        rng.shuffle(leftovers)
        if preferred_ids:
            leftovers.sort(key=lambda x: 0 if x["source_article_id"] in preferred_ids else 1)
        for item in leftovers:
            if len(picked) >= k:
                break
            if _accept(item):
                picked.append(item)

    if len(picked) < k:
        raise RuntimeError(f"Không đủ mẫu sạch để lấy {k} mẫu (chỉ tìm được {len(picked)}).")

    rng.shuffle(picked)
    return picked


def scan_news_parquet(
    parquet_path: Path,
    ai_col: str,
    generator_name: str,
    max_scan_human: int,
    max_scan_ai: int,
    rng: random.Random,
    row_slice: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Quét file Parquet của A1 với cắt đầu đối xứng (dùng cột user_prompt)."""
    pf = pq.ParquetFile(parquet_path)
    avail_cols = set(pf.schema.names)
    cols = ["id", "text", ai_col]
    has_prompt = "user_prompt" in avail_cols
    if has_prompt:
        cols.append("user_prompt")
    table = pf.read(columns=cols)
    total = table.num_rows
    lo, hi = row_slice if row_slice is not None else (0, total)
    indices = list(range(lo, min(hi, total)))
    rng.shuffle(indices)

    ids = table["id"].to_pylist()
    human_texts = table["text"].to_pylist()
    ai_texts = table[ai_col].to_pylist()
    prompts = table["user_prompt"].to_pylist() if has_prompt else [""] * total

    human_pool: List[Dict[str, Any]] = []
    ai_pool: List[Dict[str, Any]] = []

    for idx in indices:
        src_id = f"news_{ids[idx]}"
        up = prompts[idx] or ""

        if len(human_pool) < max_scan_human:
            h_trimmed = _strip_prompt_prefix(human_texts[idx] or "", up, fallback_chars=160)
            res_h = prepare_clean_doc(h_trimmed, trim_broken_leading=True)
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

        if len(ai_pool) < max_scan_ai:
            ai_raw = ai_texts[idx] or ""
            # Gate G0: nếu cột AI còn dính user_prompt ở đầu thì cắt trước khi bỏ câu cụt đầu
            if up and ai_raw.lstrip().startswith(up):
                ai_raw = ai_raw.lstrip()[len(up) :]
            res_a = prepare_clean_doc(ai_raw, trim_broken_leading=True)
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

        if len(human_pool) >= max_scan_human and len(ai_pool) >= max_scan_ai:
            break

    return human_pool, ai_pool


def scan_wikipedia_pool(wiki_dir: Path, needed: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Quét nhiều file Parquet B8 Wikipedia, bỏ câu định nghĩa mở bài để đối xứng vị trí."""
    part_files = sorted(wiki_dir.glob("viet_wiki_part_*.parquet"))
    pool: List[Dict[str, Any]] = []
    seen_wiki_ids: Set[str] = set()

    for pf_path in part_files:
        if len(pool) >= needed:
            break
        pf = pq.ParquetFile(pf_path)
        table = pf.read(columns=["page_id", "text"])
        indices = list(range(table.num_rows))
        rng.shuffle(indices)
        pids = table["page_id"].to_pylist()
        texts = table["text"].to_pylist()

        for idx in indices:
            raw = texts[idx]
            if not raw or len(raw) < 900:
                continue
            src_id = f"wiki_{pids[idx]}"
            if src_id in seen_wiki_ids:
                continue
            # Cắt ~220 ký tự đầu và bật trim_broken_leading=True để bỏ câu định nghĩa mở bài Wiki
            snippet = raw[220:3300]
            res = prepare_clean_doc(snippet, trim_broken_leading=True, min_sents=6, max_sents=24)
            if res is None:
                continue
            ft, sents, spans, nw = res
            seen_wiki_ids.add(src_id)
            pool.append(
                {
                    "source_article_id": src_id,
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
    """Quét nhiều file Parquet B9 Books, trích đoạn giữa sách sạch (bỏ bìa/mục lục)."""
    part_files = sorted((books_dir / "data").glob("train-*.parquet"))
    pool: List[Dict[str, Any]] = []

    for f_idx, pf_path in enumerate(part_files):
        if len(pool) >= needed:
            break
        pf = pq.ParquetFile(pf_path)
        table = pf.read(columns=["text"])
        indices = list(range(table.num_rows))
        rng.shuffle(indices)
        texts = table["text"].to_pylist()

        for idx in indices:
            raw = texts[idx]
            if not raw or len(raw) < 12000:
                continue
            mid_start = int(len(raw) * rng.uniform(0.25, 0.70))
            chunk = raw[mid_start : mid_start + 3100]
            if EBOOK_NOISE_RE.search(chunk):
                continue
            res = prepare_clean_doc(chunk, trim_broken_leading=True, min_sents=6, max_sents=24)
            if res is None:
                continue
            ft, sents, spans, nw = res
            pool.append(
                {
                    "source_article_id": f"book_s{f_idx}_{idx:05d}",
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
    subset: str = "core",
    eval_track: Optional[str] = None,
) -> Dict[str, Any]:
    """Đóng gói bản ghi theo schema v3: b = N cho 100% Human, b = 0 cho 100% AI."""
    sents = item["sentences"]
    n_sents = len(sents)
    rec: Dict[str, Any] = {
        "sample_id": sample_id,
        "source_article_id": item["source_article_id"],
        "split": split,
        "subset": subset,
        "is_ood_generator": is_ood_generator,
        "domain": item["domain"],
        "doc_label": doc_label,
        "doc_type": doc_type,
        "generator": generator,
        "boundary_sentence_idx": n_sents if doc_label == 0 else 0,
        "full_text": item["full_text"],
        "sentence_char_spans": item["sentence_char_spans"],
        "sentences": sents,
        "sentence_labels": [0] * n_sents if doc_label == 0 else [1] * n_sents,
    }
    if eval_track is not None:
        rec["eval_track"] = eval_track
    return rec


def make_generation_prompt(
    seed_item: Dict[str, Any],
    sample_id: str,
    split: str,
    target_doc_type: str,
    generator: str,
    is_ood_generator: bool,
    rng: random.Random,
    eval_track: Optional[str] = None,
) -> Dict[str, Any]:
    """Tạo bản ghi mồi cho vLLM (không dấu cách thừa ở cuối prefix_text)."""
    sents = seed_item["sentences"]
    n = len(sents)
    if target_doc_type == "mixed_human_ai":
        lo = max(2, int(n * 0.20))
        hi = min(n - 2, max(lo, int(n * 0.80)))
        k = rng.randint(lo, hi)
        prefix_sents = sents[:k]
        prefix_words = sum(len(s.split()) for s in prefix_sents)
        tail_words = max(60, seed_item["n_words"] - prefix_words)
        target_words = max(60, min(380, tail_words))
        max_new_tokens = min(520, max(120, int(target_words * 1.45) + 35))
    else:
        k = 2
        prefix_sents = sents[:2]
        target_words = max(160, min(420, seed_item["n_words"]))
        max_new_tokens = min(700, max(220, int(target_words * 1.55) + 80))

    prefix_text = " ".join(prefix_sents).rstrip()
    rec: Dict[str, Any] = {
        "sample_id": sample_id,
        "source_article_id": seed_item["source_article_id"],
        "split": split,
        "subset": "core",
        "is_ood_generator": is_ood_generator,
        "domain": seed_item["domain"],
        "target_doc_type": target_doc_type,
        "generator": generator,
        "bucket_key": f"{split}:{target_doc_type}:{generator}:{seed_item['domain']}",
        "cut_sentence_idx": k,
        "drop_prefix_after_gen": target_doc_type == "ai",
        "prefix_sentences": prefix_sents,
        "prefix_text": prefix_text,
        "orig_total_words": seed_item["n_words"],
        "orig_total_sents": n,
        "target_new_words": target_words,
        "max_new_tokens": max_new_tokens,
    }
    if eval_track is not None:
        rec["eval_track"] = eval_track
    return rec


def build_stress_tests(
    datasets_dir: Path,
    out_dir: Path,
    lsh_index: MinHashLSHIndex,
    used_fps: Set[str],
    rng: random.Random,
) -> Dict[str, int]:
    """Xây dựng 3 bộ Stress-Test Mở Rộng (Test-only, cắt đầu đối xứng, lọc MinHash)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}

    # 1. Stress-Test 1 (P0): OOD Miền Văn Học Vũ Trọng Phụng (A2)
    s_pf = pq.ParquetFile(datasets_dir / "vu-trong-phung/Sailor2-8B-VuTrongPhung.parquet").read()
    g_pf = pq.ParquetFile(datasets_dir / "vu-trong-phung/Gemma-3-12B-VuTrongPhung.parquet").read()

    vtp_records: List[Dict[str, Any]] = []
    seen_vtp_human: Set[str] = set()

    for row in s_pf.to_pylist():
        src_id = f"vtp_{row['id'][:16]}"
        if src_id not in seen_vtp_human:
            raw_h = (row["text"] or "")[140:]
            res_h = prepare_clean_doc(raw_h, trim_broken_leading=True, min_sents=4, min_words=80)
            if res_h is not None:
                ft, sents, spans, _ = res_h
                if not lsh_index.query_duplicates(ft):
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
                            subset="stress",
                        )
                    )

        res_s = prepare_clean_doc(
            row["sail-Sailor2-8B-Chat-hf_generated_text_wo_prompt"],
            trim_broken_leading=True,
            min_sents=4,
            min_words=80,
        )
        if res_s is not None:
            ft, sents, spans, _ = res_s
            if not lsh_index.query_duplicates(ft):
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
                        subset="stress",
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
            ft, sents, spans, _ = res_g
            if not lsh_index.query_duplicates(ft):
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
                        subset="stress",
                    )
                )

    write_jsonl(out_dir / "test_vtp_domain.jsonl", vtp_records)
    stats["stress_vtp_domain"] = len(vtp_records)

    # 2. Stress-Test 2 (P0): Commercial LLMs (A5 - ICCIES 2025: 500 Human + 500 AI, MinHash dedup)
    iccies_human: List[Dict[str, Any]] = []
    iccies_ai: List[Dict[str, Any]] = []
    seen_iccies_fp: Set[str] = set(used_fps)

    human_candidates: List[Tuple[int, str]] = []
    ai_candidates: List[Tuple[int, str]] = []
    reservoir_size = 4000
    with open(datasets_dir / "iccies-2025-news-human-ai/Ai_human_200k.csv", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            lbl = row["Label"].strip()
            txt = row["Text"]
            if lbl == "0":
                if len(human_candidates) < reservoir_size:
                    human_candidates.append((row_idx, txt))
                else:
                    j = rng.randint(0, row_idx)
                    if j < reservoir_size:
                        human_candidates[j] = (row_idx, txt)
            elif lbl == "1":
                if len(ai_candidates) < reservoir_size:
                    ai_candidates.append((row_idx, txt))
                else:
                    j = rng.randint(0, row_idx)
                    if j < reservoir_size:
                        ai_candidates[j] = (row_idx, txt)

    rng.shuffle(human_candidates)
    rng.shuffle(ai_candidates)

    for row_idx, txt in human_candidates:
        if len(iccies_human) >= 500:
            break
        raw_h = (txt or "")[140:]
        res = prepare_clean_doc(raw_h, trim_broken_leading=True, min_sents=5, min_words=100)
        if res is None:
            continue
        ft, sents, spans, _ = res
        fp = dedup_fingerprint(ft)
        if fp in seen_iccies_fp or lsh_index.query_duplicates(ft):
            continue
        seen_iccies_fp.add(fp)
        sid = f"iccies_h_{row_idx}"
        lsh_index.add(sid, ft)
        iccies_human.append(
            make_ready_record(
                {
                    "source_article_id": sid,
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
                subset="stress",
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
        if fp in seen_iccies_fp or lsh_index.query_duplicates(ft):
            continue
        seen_iccies_fp.add(fp)
        sid = f"iccies_a_{row_idx}"
        lsh_index.add(sid, ft)
        iccies_ai.append(
            make_ready_record(
                {
                    "source_article_id": sid,
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
                subset="stress",
            )
        )

    iccies_all = iccies_human + iccies_ai
    rng.shuffle(iccies_all)
    write_jsonl(out_dir / "test_iccies_commercial.jsonl", iccies_all)
    stats["stress_iccies_commercial"] = len(iccies_all)

    # 3. Stress-Test 3 (P1): Paraphrase & Humanizer Robustness (A4 - Text-Rewriting)
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
                    raw_h = (obj["text"] or "")[140:]
                    res_h = prepare_clean_doc(raw_h, trim_broken_leading=True, min_sents=4, min_words=80)
                    if res_h is not None:
                        seen_rw_human.add(fp_h)
                        ft, sents, spans, _ = res_h
                        if not lsh_index.query_duplicates(ft):
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
                                subset="stress",
                            )
                            rec["rewrite_stage"] = "human"
                            rewrite_records.append(rec)

                for col_key, stage_name in variant_cols:
                    raw_val = obj.get(col_key, "")
                    res_v = prepare_clean_doc(raw_val, trim_broken_leading=True, min_sents=4, min_words=70)
                    if res_v is None:
                        continue
                    ft, sents, spans, _ = res_v
                    if lsh_index.query_duplicates(ft):
                        continue
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
                        subset="stress",
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


def _load_existing_gen_by_bucket(gen_path: Path) -> Dict[str, Set[str]]:
    """Đọc các source_article_id đã sinh trước đó (nếu có) theo bucket để ưu tiên tái sử dụng."""
    by_bucket: Dict[str, Set[str]] = collections.defaultdict(set)
    if not gen_path.exists():
        return by_bucket
    with open(gen_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            b_key = f"{obj.get('split')}:{obj.get('target_doc_type')}:{obj.get('generator')}:{obj.get('domain')}"
            if obj.get("source_article_id"):
                by_bucket[b_key].add(obj["source_article_id"])
    return by_bucket


def main() -> None:
    rng = random.Random(SEED)
    datasets_dir = PROJECT_ROOT / "datasets"
    interim_dir = PROJECT_ROOT / "data" / "interim"
    calib_dir = PROJECT_ROOT / "data" / "calib"
    stress_dir = PROJECT_ROOT / "data" / "stress_test"

    existing_gen_buckets = _load_existing_gen_by_bucket(interim_dir / "generated_continuations.jsonl")

    print("[1/6] Quét & cắt đầu đối xứng bể ứng viên từ A1, B8 (Wiki), B9 (Books)...", flush=True)
    train_news_h, train_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/train.parquet",
        ai_col="sail-Sailor2-8B-Chat-generated_text",
        generator_name="sail/Sailor2-8B-Chat",
        max_scan_human=6000,
        max_scan_ai=1600,
        rng=rng,
    )
    val_pf_rows = pq.ParquetFile(datasets_dir / "vietnamese-news/validation.parquet").metadata.num_rows
    mid_val = int(val_pf_rows * 0.60)
    val_news_h, val_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/validation.parquet",
        ai_col="sail-Sailor2-8B-Chat-generated_text",
        generator_name="sail/Sailor2-8B-Chat",
        max_scan_human=2600,
        max_scan_ai=600,
        rng=rng,
        row_slice=(0, mid_val),
    )
    test_id_news_h, test_id_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/validation.parquet",
        ai_col="sail-Sailor2-8B-Chat-generated_text",
        generator_name="sail/Sailor2-8B-Chat",
        max_scan_human=1800,
        max_scan_ai=400,
        rng=rng,
        row_slice=(mid_val, val_pf_rows),
    )
    test_ood_news_h, test_ood_news_ai = scan_news_parquet(
        datasets_dir / "vietnamese-news/test.parquet",
        ai_col="google-gemma-3-12b-it-generated_text",
        generator_name="google/gemma-3-12b-it",
        max_scan_human=2200,
        max_scan_ai=700,
        rng=rng,
    )

    # Bể Wiki (1,200) & Books (800) sau khi cắt đầu & length_ok
    wiki_pool = scan_wikipedia_pool(datasets_dir / "vietnamese-wikipedia-clean", needed=1200, rng=rng)
    books_pool = scan_books_pool(datasets_dir / "vietnamese-book-corpus", needed=800, rng=rng)
    if len(wiki_pool) < 1200 or len(books_pool) < 800:
        raise RuntimeError(f"Thiếu bể Wiki ({len(wiki_pool)}/1200) hoặc Books ({len(books_pool)}/800)!")

    wiki_train, wiki_val, wiki_test = wiki_pool[:700], wiki_pool[700:900], wiki_pool[900:1200]
    books_train, books_val, books_test = books_pool[:450], books_pool[450:600], books_pool[600:800]

    print(
        f"   -> Bể sạch: TrainNewsH={len(train_news_h)}, TrainNewsAI={len(train_news_ai)}, "
        f"ValNewsH={len(val_news_h)}, TestIDNewsH={len(test_id_news_h)}, TestOODNewsH={len(test_ood_news_h)}, "
        f"Wiki={len(wiki_pool)}, Books={len(books_pool)}",
        flush=True,
    )

    print("[2/6] Rút tập Core (Bảng 3.3, +20% mồi sinh = 3,300 prompts) & MinHash dedup...", flush=True)
    used_ids: Set[str] = set()
    used_fps: Set[str] = set()
    lsh_index = MinHashLSHIndex(threshold=0.8)

    # Tránh đưa các ID đang nằm trong generated_continuations cũ vào Human/AI ready hoặc Calib
    all_old_gen_ids: Set[str] = set()
    for s in existing_gen_buckets.values():
        all_old_gen_ids.update(s)

    def _non_gen_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not all_old_gen_ids:
            return items
        return [x for x in items if x["source_article_id"] not in all_old_gen_ids]

    ready_records: List[Dict[str, Any]] = []
    prompt_records: List[Dict[str, Any]] = []

    def _add_prompts(
        news_pool: List[Dict[str, Any]],
        wiki_sub: List[Dict[str, Any]],
        books_sub: List[Dict[str, Any]],
        counts_by_domain: Dict[str, int],
        split: str,
        target_doc_type: str,
        generator: str,
        is_ood: bool,
        id_prefix: str,
        eval_track: Optional[str] = None,
    ) -> None:
        pools = {"news": news_pool, "wikipedia": wiki_sub, "books": books_sub}
        picked_all: List[Dict[str, Any]] = []
        for dom, cnt in counts_by_domain.items():
            if cnt <= 0:
                continue
            b_key = f"{split}:{target_doc_type}:{generator}:{dom}"
            pref = existing_gen_buckets.get(b_key)
            picked_all.extend(
                stratified_pick(
                    pools[dom],
                    cnt,
                    rng,
                    used_ids,
                    used_fps,
                    lsh_index=lsh_index,
                    preferred_ids=pref,
                )
            )
        rng.shuffle(picked_all)
        for i, item in enumerate(picked_all):
            prompt_records.append(
                make_generation_prompt(
                    item,
                    f"{id_prefix}_{i:04d}",
                    split,
                    target_doc_type,
                    generator,
                    is_ood,
                    rng,
                    eval_track=eval_track,
                )
            )

    # ================= TRAIN (Đích 3,500: Ready 1,780 + Prompts 2,064 [+20% cho 1,720]) =================
    tr_h = (
        stratified_pick(_non_gen_pool(train_news_h), 850, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(wiki_train), 100, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(books_train), 50, rng, used_ids, used_fps, lsh_index)
    )
    for i, item in enumerate(tr_h):
        ready_records.append(make_ready_record(item, f"train_human_{i:04d}", "train", 0, "human", "human", False))

    tr_ai_sailor = stratified_pick(train_news_ai, 780, rng, used_ids, used_fps, lsh_index)
    for i, item in enumerate(tr_ai_sailor):
        ready_records.append(
            make_ready_record(item, f"train_ai_sailor2_{i:04d}", "train", 1, "ai", item["generator"], False)
        )

    # Train AI-Qwen: đích 220 (160/40/20) -> +20% = 264 (192/48/24)
    _add_prompts(
        train_news_h, wiki_train, books_train,
        {"news": 192, "wikipedia": 48, "books": 24},
        "train", "ai", "Qwen/Qwen2.5-7B", False, "train_ai_qwen",
    )
    # Train Mixed-Qwen: đích 1,500 (1,250/150/100) -> +20% = 1,800 (1,500/180/120)
    _add_prompts(
        train_news_h, wiki_train, books_train,
        {"news": 1500, "wikipedia": 180, "books": 120},
        "train", "mixed_human_ai", "Qwen/Qwen2.5-7B", False, "train_mixed_qwen",
    )

    # ================= VAL (Đích 750: Ready 370 + Prompts 456 [+20% cho 380]) =================
    va_h = (
        stratified_pick(_non_gen_pool(val_news_h), 185, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(wiki_val), 20, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(books_val), 15, rng, used_ids, used_fps, lsh_index)
    )
    for i, item in enumerate(va_h):
        ready_records.append(make_ready_record(item, f"val_human_{i:04d}", "val", 0, "human", "human", False))

    va_ai_sailor = stratified_pick(val_news_ai, 150, rng, used_ids, used_fps, lsh_index)
    for i, item in enumerate(va_ai_sailor):
        ready_records.append(
            make_ready_record(item, f"val_ai_sailor2_{i:04d}", "val", 1, "ai", item["generator"], False)
        )

    # Val AI-Qwen: đích 50 (35/10/5) -> +20% = 60 (42/12/6)
    _add_prompts(
        val_news_h, wiki_val, books_val,
        {"news": 42, "wikipedia": 12, "books": 6},
        "val", "ai", "Qwen/Qwen2.5-7B", False, "val_ai_qwen",
    )
    # Val Mixed-Qwen: đích 330 (280/30/20) -> +20% = 396 (336/36/24)
    _add_prompts(
        val_news_h, wiki_val, books_val,
        {"news": 336, "wikipedia": 36, "books": 24},
        "val", "mixed_human_ai", "Qwen/Qwen2.5-7B", False, "val_mixed_qwen",
    )

    # ================= TEST-ID (Đích 390: Ready 240 + Prompts 180 [+20% cho 150]) =================
    te_id_h = (
        stratified_pick(_non_gen_pool(test_id_news_h), 120, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(wiki_test), 30, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(books_test), 20, rng, used_ids, used_fps, lsh_index)
    )
    for i, item in enumerate(te_id_h):
        ready_records.append(
            make_ready_record(item, f"test_id_human_{i:04d}", "test", 0, "human", "human", False, eval_track="test_id")
        )

    te_ai_sailor = stratified_pick(test_id_news_ai, 70, rng, used_ids, used_fps, lsh_index)
    for i, item in enumerate(te_ai_sailor):
        ready_records.append(
            make_ready_record(
                item, f"test_ai_sailor2_{i:04d}", "test", 1, "ai", item["generator"], False, eval_track="test_id"
            )
        )

    # Test-ID AI-Qwen: đích 30 (20/5/5) -> +20% = 36 (24/6/6)
    _add_prompts(
        test_id_news_h, wiki_test, books_test,
        {"news": 24, "wikipedia": 6, "books": 6},
        "test", "ai", "Qwen/Qwen2.5-7B", False, "test_ai_qwen", eval_track="test_id",
    )
    # Test-ID Mixed-Qwen: đích 120 (95/15/10) -> +20% = 144 (114/18/12)
    _add_prompts(
        test_id_news_h, wiki_test, books_test,
        {"news": 114, "wikipedia": 18, "books": 12},
        "test", "mixed_human_ai", "Qwen/Qwen2.5-7B", False, "test_mixed_qwen", eval_track="test_id",
    )

    # ================= TEST-OOD (Đích 850: Ready 350 + Prompts 600 [+20% cho 500]) =================
    te_ood_h = (
        stratified_pick(_non_gen_pool(test_ood_news_h), 110, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(wiki_test), 25, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(books_test), 15, rng, used_ids, used_fps, lsh_index)
    )
    for i, item in enumerate(te_ood_h):
        ready_records.append(
            make_ready_record(
                item, f"test_ood_human_{i:04d}", "test", 0, "human", "human", False, eval_track="test_ood"
            )
        )

    te_ai_gemma = stratified_pick(test_ood_news_ai, 200, rng, used_ids, used_fps, lsh_index)
    for i, item in enumerate(te_ai_gemma):
        ready_records.append(
            make_ready_record(
                item, f"test_ai_gemma3_{i:04d}", "test", 1, "ai", item["generator"], True, eval_track="test_ood"
            )
        )

    # Test-OOD AI-Llama: đích 100 (70/20/10) -> +20% = 120 (84/24/12)
    _add_prompts(
        test_ood_news_h, wiki_test, books_test,
        {"news": 84, "wikipedia": 24, "books": 12},
        "test", "ai", "meta-llama/Llama-3.1-8B", True, "test_ai_llama", eval_track="test_ood",
    )
    # Test-OOD Mixed-Llama: đích 400 (330/45/25) -> +20% = 480 (396/54/30)
    _add_prompts(
        test_ood_news_h, wiki_test, books_test,
        {"news": 396, "wikipedia": 54, "books": 30},
        "test", "mixed_human_ai", "meta-llama/Llama-3.1-8B", True, "test_mixed_llama", eval_track="test_ood",
    )

    print("[3/6] Rút Human Calibration Pool (1,000 mẫu: 500 Val-Calib + 500 Test-Calib)...", flush=True)
    val_calib_items = (
        stratified_pick(_non_gen_pool(val_news_h), 400, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(wiki_val), 60, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(books_val), 40, rng, used_ids, used_fps, lsh_index)
    )
    val_calib_records = [
        make_ready_record(item, f"val_calib_{i:04d}", "val_calib", 0, "human", "human", False, subset="calib")
        for i, item in enumerate(val_calib_items)
    ]

    test_calib_items = (
        stratified_pick(_non_gen_pool(test_id_news_h), 300, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(test_ood_news_h), 100, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(wiki_test), 60, rng, used_ids, used_fps, lsh_index)
        + stratified_pick(_non_gen_pool(books_test), 40, rng, used_ids, used_fps, lsh_index)
    )
    test_calib_records = [
        make_ready_record(item, f"test_calib_{i:04d}", "test_calib", 0, "human", "human", False, subset="calib")
        for i, item in enumerate(test_calib_items)
    ]

    write_jsonl(calib_dir / "val_calib.jsonl", val_calib_records)
    write_jsonl(calib_dir / "test_calib.jsonl", test_calib_records)

    print("[4/6] Ghi `core_ready_part1.jsonl` (2,740), `prompts_to_generate.jsonl` (3,300) & `sample_debug.jsonl`...", flush=True)
    write_jsonl(interim_dir / "core_ready_part1.jsonl", ready_records)
    write_jsonl(interim_dir / "prompts_to_generate.jsonl", prompt_records)

    # Tạo `data/sample_debug.jsonl` (20 mẫu từ train/val, KHÔNG chứa mẫu test thật để giữ kỷ luật test)
    debug_samples = (
        [r for r in ready_records if r["split"] == "train" and r["doc_type"] == "human"][:6]
        + [r for r in ready_records if r["split"] == "train" and r["doc_type"] == "ai"][:6]
        + [r for r in ready_records if r["split"] == "val" and r["doc_type"] == "human"][:3]
    )
    train_h_list = [r for r in ready_records if r["split"] == "train" and r["doc_type"] == "human"]
    train_a_list = [r for r in ready_records if r["split"] == "train" and r["doc_type"] == "ai"]
    for d_idx in range(5):
        h_rec = train_h_list[d_idx]
        a_rec = train_a_list[d_idx]
        k_cut = max(2, len(h_rec["sentences"]) // 2)
        mix_sents = h_rec["sentences"][:k_cut] + a_rec["sentences"][: max(3, len(a_rec["sentences"]) // 2)]
        ft_m, sp_m = build_text_and_spans(mix_sents)
        debug_samples.append(
            {
                "sample_id": f"debug_mixed_mock_{d_idx:02d}",
                "source_article_id": f"debug_src_{d_idx:02d}",
                "split": "train" if d_idx < 3 else "val",
                "subset": "debug",
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

    print("[5/6] Tạo 3 bộ Stress-Test Mở Rộng (`data/stress_test/`, cắt đầu đối xứng & MinHash dedup)...", flush=True)
    stress_stats = build_stress_tests(datasets_dir, stress_dir, lsh_index, used_fps, rng)

    print("[6/6] Kiểm tra toàn vẹn Zero Leakage giữa mọi split và Calib...", flush=True)
    all_src_by_split: Dict[str, Set[str]] = collections.defaultdict(set)
    for r in ready_records:
        all_src_by_split[r["split"]].add(r["source_article_id"])
    for p in prompt_records:
        all_src_by_split[p["split"]].add(p["source_article_id"])
    for c in val_calib_records:
        all_src_by_split["val_calib"].add(c["source_article_id"])
    for c in test_calib_records:
        all_src_by_split["test_calib"].add(c["source_article_id"])

    group_names = ["train", "val", "test", "val_calib", "test_calib"]
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1 :]:
            overlap = all_src_by_split[g1] & all_src_by_split[g2]
            assert not overlap, f"Rò rỉ source_article_id giữa {g1} và {g2}: {len(overlap)} mẫu!"

    summary = {
        "core_ready_count": len(ready_records),
        "prompts_to_generate_count": len(prompt_records),
        "val_calib_count": len(val_calib_records),
        "test_calib_count": len(test_calib_records),
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

    print("HOÀN TẤT PHẦN 1 (v3)!", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
