"""Ghép kết quả `generated_continuations.jsonl` (từ Kaggle vLLM) với `core_ready_part1.jsonl`
để tạo bộ dữ liệu Core chuẩn v3 trong `data/processed/`:
  - `data/processed/train.jsonl` (mục tiêu 3,500 mẫu)
  - `data/processed/val.jsonl` (mục tiêu 750 mẫu)
  - `data/processed/test.jsonl` (mục tiêu 1,240 mẫu: 390 ID + 850 OOD)
  - `data/processed/sample_500.jsonl` (400 train + 100 val; KHÔNG chứa mẫu test)
  - `data/processed/dataset_stats.json`
  - `data/interim/prompts_delta_to_generate.jsonl` (chỉ các mồi còn thiếu để chạy bù trên Kaggle nếu cần)
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.text_utils import (
    MinHashLSHIndex,
    build_text_and_spans,
    clean_generated_continuation,
    is_clean_vietnamese,
    length_ok,
    normalize_text_basic,
    split_sentences,
)

# Đích số lượng sinh theo Bảng 3.3 (tổng 2,750 mẫu sinh + 2,740 ready = 5,490 mẫu Core)
TARGET_GEN_COUNTS: Dict[Tuple[str, str, str, str], int] = {
    ("train", "ai", "Qwen/Qwen2.5-7B", "news"): 160,
    ("train", "ai", "Qwen/Qwen2.5-7B", "wikipedia"): 40,
    ("train", "ai", "Qwen/Qwen2.5-7B", "books"): 20,
    ("train", "mixed_human_ai", "Qwen/Qwen2.5-7B", "news"): 1250,
    ("train", "mixed_human_ai", "Qwen/Qwen2.5-7B", "wikipedia"): 150,
    ("train", "mixed_human_ai", "Qwen/Qwen2.5-7B", "books"): 100,
    ("val", "ai", "Qwen/Qwen2.5-7B", "news"): 35,
    ("val", "ai", "Qwen/Qwen2.5-7B", "wikipedia"): 10,
    ("val", "ai", "Qwen/Qwen2.5-7B", "books"): 5,
    ("val", "mixed_human_ai", "Qwen/Qwen2.5-7B", "news"): 280,
    ("val", "mixed_human_ai", "Qwen/Qwen2.5-7B", "wikipedia"): 30,
    ("val", "mixed_human_ai", "Qwen/Qwen2.5-7B", "books"): 20,
    ("test", "ai", "Qwen/Qwen2.5-7B", "news"): 20,
    ("test", "ai", "Qwen/Qwen2.5-7B", "wikipedia"): 5,
    ("test", "ai", "Qwen/Qwen2.5-7B", "books"): 5,
    ("test", "mixed_human_ai", "Qwen/Qwen2.5-7B", "news"): 95,
    ("test", "mixed_human_ai", "Qwen/Qwen2.5-7B", "wikipedia"): 15,
    ("test", "mixed_human_ai", "Qwen/Qwen2.5-7B", "books"): 10,
    ("test", "ai", "meta-llama/Llama-3.1-8B", "news"): 70,
    ("test", "ai", "meta-llama/Llama-3.1-8B", "wikipedia"): 20,
    ("test", "ai", "meta-llama/Llama-3.1-8B", "books"): 10,
    ("test", "mixed_human_ai", "meta-llama/Llama-3.1-8B", "news"): 330,
    ("test", "mixed_human_ai", "meta-llama/Llama-3.1-8B", "wikipedia"): 45,
    ("test", "mixed_human_ai", "meta-llama/Llama-3.1-8B", "books"): 25,
}


def _length_bin(num_words: int) -> str:
    if num_words < 240:
        return "short"
    if num_words < 400:
        return "medium"
    return "long"


def _stratified_downsample(items: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(items) <= k:
        return list(items)
    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for it in items:
        nw = len(it["full_text"].split())
        buckets[_length_bin(nw)].append(it)
    for b in buckets.values():
        rng.shuffle(b)

    targets = {"short": int(round(k * 0.30)), "medium": int(round(k * 0.45))}
    targets["long"] = k - targets["short"] - targets["medium"]

    picked: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for b_name in ("short", "medium", "long"):
        b_list = buckets[b_name]
        need = targets[b_name]
        picked.extend(b_list[:need])
        leftovers.extend(b_list[need:])

    if len(picked) < k:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: k - len(picked)])
    rng.shuffle(picked)
    return picked[:k]


def finalize_generated_record(raw_rec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Hậu xử lý phần văn bản do LLM sinh tiếp:
    - Cắt tại \\n\\n / dòng lạc đề, chuẩn hóa dấu câu đồng nhất cho cả prefix và continuation.
    - Kiểm định tiếng Việt riêng trên phần AI.
    - Cân chỉnh câu AI dư nếu tổng độ dài vượt khung `length_ok()` (6-24 câu, 150-650 từ).
    """
    raw_prefix_sents: List[str] = raw_rec.get("prefix_sentences", [])
    prefix_sents: List[str] = []
    for s in raw_prefix_sents:
        s_norm = normalize_text_basic(s)
        if s_norm:
            prefix_sents.extend(split_sentences(s_norm) or [s_norm])

    is_pure_ai = bool(raw_rec.get("drop_prefix_after_gen", False))
    is_legacy_run = "orig_total_words" not in raw_rec

    # Nếu bản ghi đến từ lần sinh cũ chưa cắt câu mở bài, bỏ câu đầu của prefix Mixed để đối xứng vị trí
    if is_legacy_run and not is_pure_ai and len(prefix_sents) >= 3:
        prefix_sents = prefix_sents[1:]

    raw_cont: str = raw_rec.get("raw_continuation", "")
    cleaned_cont = clean_generated_continuation(raw_cont)
    if not cleaned_cont:
        return None, "empty_or_forbidden_after_clean"

    ai_sents = split_sentences(cleaned_cont)
    if ai_sents and prefix_sents and ai_sents[0].strip().lower() == prefix_sents[-1].strip().lower():
        ai_sents = ai_sents[1:]

    if is_legacy_run:
        # Lần sinh cũ nối dấu cách cuối prompt khiến 57.4% câu AI đầu bắt đầu bằng token "Ở "
        if not is_pure_ai and len(ai_sents) >= 3 and ai_sents[0].startswith("Ở "):
            ai_sents = ai_sents[1:]
        if (not is_pure_ai and len(ai_sents) >= 3) or (is_pure_ai and len(ai_sents) >= 8):
            ai_sents = ai_sents[:-1]

    min_ai_sents = 6 if is_pure_ai else 2
    min_ai_words = 150 if is_pure_ai else 22

    if len(ai_sents) < min_ai_sents:
        return None, f"too_few_ai_sents(<{min_ai_sents})"

    # Cắt bớt câu AI dư ở cuối nếu vượt quá trần 24 câu / 650 từ (hoặc vượt quá độ dài bài gốc mục tiêu)
    orig_words = int(raw_rec.get("orig_total_words") or 430)
    word_cap = min(650, max(360, orig_words + 60)) if not is_pure_ai else 650

    if is_pure_ai:
        while len(ai_sents) > 24 or (len(ai_sents) > min_ai_sents and len(" ".join(ai_sents).split()) > word_cap):
            ai_sents = ai_sents[:-1]
    else:
        if len(prefix_sents) < 2 or len(prefix_sents) > 22:
            return None, "prefix_sents_out_of_range"
        while (len(prefix_sents) + len(ai_sents) > 24) or (
            len(ai_sents) > min_ai_sents
            and len(" ".join(prefix_sents + ai_sents).split()) > word_cap
        ):
            ai_sents = ai_sents[:-1]

    if len(ai_sents) < min_ai_sents:
        return None, f"too_few_ai_sents_after_trim(<{min_ai_sents})"

    ai_joined = " ".join(ai_sents)
    if not is_clean_vietnamese(ai_joined, min_words=min_ai_words, max_words=650):
        return None, "ai_part_failed_clean_vi_check"

    if is_pure_ai:
        final_sents = ai_sents
        sent_labels = [1] * len(final_sents)
        boundary_idx = 0
        doc_type = "ai"
    else:
        k = len(prefix_sents)
        final_sents = prefix_sents + ai_sents
        sent_labels = [0] * k + [1] * len(ai_sents)
        boundary_idx = k
        doc_type = "mixed_human_ai"

    full_text, spans = build_text_and_spans(final_sents)
    if not length_ok(final_sents, full_text, min_sents=6, max_sents=24, min_words=150, max_words=650):
        return None, "failed_length_ok(6-24s,150-650w)"

    rec: Dict[str, Any] = {
        "sample_id": raw_rec["sample_id"],
        "source_article_id": raw_rec["source_article_id"],
        "split": raw_rec["split"],
        "subset": "core",
        "is_ood_generator": bool(raw_rec["is_ood_generator"]),
        "domain": raw_rec["domain"],
        "doc_label": 1,
        "doc_type": doc_type,
        "generator": raw_rec["generator"],
        "boundary_sentence_idx": boundary_idx,
        "full_text": full_text,
        "sentence_char_spans": spans,
        "sentences": final_sents,
        "sentence_labels": sent_labels,
    }
    if "gen_config" in raw_rec:
        rec["gen_config"] = raw_rec["gen_config"]
    if "eval_track" in raw_rec:
        rec["eval_track"] = raw_rec["eval_track"]
    elif raw_rec["split"] == "test":
        rec["eval_track"] = "test_ood" if raw_rec["is_ood_generator"] else "test_id"
    return rec, "ok"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_all_splits(by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    """Kiểm định nghiêm ngặt schema v3, bất biến ranh giới b in {0..N} và zero leakage."""
    seen_sample_ids: Set[str] = set()
    src_by_split: Dict[str, Set[str]] = {sp: set() for sp in by_split}

    for sp, rows in by_split.items():
        for r in rows:
            sid = r["sample_id"]
            assert sid not in seen_sample_ids, f"Trùng sample_id: {sid}"
            seen_sample_ids.add(sid)
            src_by_split[sp].add(r["source_article_id"])

            sents = r["sentences"]
            spans = r["sentence_char_spans"]
            labels = r["sentence_labels"]
            ft = r["full_text"]
            n_sents = len(sents)
            assert n_sents == len(spans) == len(labels), f"Lỗi độ dài mảng câu ở {sid}"
            assert length_ok(sents, ft, min_sents=6, max_sents=24, min_words=150, max_words=650), (
                f"Vi phạm length_ok ở {sid}: {n_sents} sents, {len(ft.split())} words"
            )
            for (st, ed), s in zip(spans, sents):
                assert ft[st:ed] == s, f"Lệch span ở {sid}"

            b = r["boundary_sentence_idx"]
            if r["doc_type"] == "human":
                assert r["doc_label"] == 0 and b == n_sents, f"Human phải có b=N ở {sid}, gặp {b}"
            elif r["doc_type"] == "ai":
                assert r["doc_label"] == 1 and b == 0, f"AI thuần phải có b=0 ở {sid}, gặp {b}"
            elif r["doc_type"] == "mixed_human_ai":
                assert r["doc_label"] == 1 and 2 <= b <= n_sents - 2, f"Mixed b={b} không hợp lệ ở {sid}"
            else:
                raise AssertionError(f"doc_type lạ: {r['doc_type']}")
            assert labels == [int(i >= b) for i in range(n_sents)], f"Vi phạm bất biến sentence_labels ở {sid}"

    assert not (src_by_split["train"] & src_by_split["val"]), "Rò rỉ source_id Train <-> Val!"
    assert not (src_by_split["train"] & src_by_split["test"]), "Rò rỉ source_id Train <-> Test!"
    assert not (src_by_split["val"] & src_by_split["test"]), "Rò rỉ source_id Val <-> Test!"


def _load_calib_ids_and_lsh(calib_dir: Path) -> Tuple[Set[str], MinHashLSHIndex]:
    ids: Set[str] = set()
    lsh = MinHashLSHIndex(threshold=0.8)
    for fname in ("val_calib.jsonl", "test_calib.jsonl"):
        fpath = calib_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    ids.add(r["source_article_id"])
                    lsh.add(r["source_article_id"], r["full_text"])
    return ids, lsh


def main() -> None:
    interim_dir = PROJECT_ROOT / "data/interim"
    v2_files: List[Path] = []
    if (interim_dir / "generated_continuations_ver2.jsonl").exists():
        v2_files.append(interim_dir / "generated_continuations_ver2.jsonl")
    v2_files.extend(sorted(interim_dir.glob("generated_delta_ver*.jsonl")))
    default_gens = v2_files if v2_files else [interim_dir / "generated_continuations.jsonl"]

    parser = argparse.ArgumentParser(description="Ghép dữ liệu Phần 1 và kết quả vLLM thành train/val/test.jsonl (v3).")
    parser.add_argument(
        "--ready",
        type=Path,
        default=PROJECT_ROOT / "data/interim/core_ready_part1.jsonl",
        help="File core_ready_part1.jsonl",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        nargs="+",
        default=default_gens,
        help="Một hoặc nhiều file generated_continuations*.jsonl từ Kaggle",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROJECT_ROOT / "data/interim/prompts_to_generate.jsonl",
        help="File prompts_to_generate.jsonl (+20%%) để tính delta còn thiếu",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calib",
        help="Thư mục chứa val_calib.jsonl và test_calib.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Thư mục đầu ra data/processed",
    )
    args = parser.parse_args()

    for gen_path in args.generated:
        if not gen_path.exists():
            raise SystemExit(f"Chưa tìm thấy file kết quả vLLM tại: {gen_path}")

    rng = random.Random(42)
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    reserved_src_ids, lsh = _load_calib_ids_and_lsh(args.calib_dir)
    seen_sample_ids: Set[str] = set()

    with open(args.ready, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_split[rec["split"]].append(rec)
            reserved_src_ids.add(rec["source_article_id"])
            seen_sample_ids.add(rec["sample_id"])
            lsh.add(rec["source_article_id"], rec["full_text"])

    drop_reasons: collections.Counter[str] = collections.Counter()
    by_bucket: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    used_gen_src_ids: Set[str] = set()

    for gen_path in args.generated:
        with open(gen_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_rec = json.loads(line)
                src_id = raw_rec.get("source_article_id", "")
                if src_id in reserved_src_ids or src_id in used_gen_src_ids:
                    drop_reasons["duplicate_source_id_with_ready_or_calib"] += 1
                    continue
                fin, reason = finalize_generated_record(raw_rec)
                if fin is None:
                    drop_reasons[reason] += 1
                    continue
                if lsh.query_duplicates(fin["full_text"]):
                    drop_reasons["minhash_near_duplicate"] += 1
                    continue
                while fin["sample_id"] in seen_sample_ids:
                    fin["sample_id"] = f"{fin['sample_id']}_dup"
                seen_sample_ids.add(fin["sample_id"])
                used_gen_src_ids.add(src_id)
                lsh.add(src_id, fin["full_text"])
                b_key = (fin["split"], fin["doc_type"], fin["generator"], fin["domain"])
                by_bucket[b_key].append(fin)

    kept_gen = 0
    bucket_deficits: Dict[Tuple[str, str, str, str], int] = {}
    for b_key, target_cnt in TARGET_GEN_COUNTS.items():
        cands = by_bucket.get(b_key, [])
        if len(cands) > target_cnt:
            chosen = _stratified_downsample(cands, target_cnt, rng)
        else:
            chosen = cands
            if len(cands) < target_cnt:
                bucket_deficits[b_key] = target_cnt - len(cands)
        kept_gen += len(chosen)
        for rec in chosen:
            by_split[rec["split"]].append(rec)

    # Xuất prompts_delta_to_generate.jsonl cho những bucket còn thiếu (nếu chạy bù từ kết quả cũ)
    delta_prompts: List[Dict[str, Any]] = []
    if args.prompts.exists() and bucket_deficits:
        prompts_by_bucket: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
        all_used_ids = reserved_src_ids | used_gen_src_ids
        with open(args.prompts, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                if p["source_article_id"] in all_used_ids:
                    continue
                pk = (p["split"], p["target_doc_type"], p["generator"], p["domain"])
                prompts_by_bucket[pk].append(p)
        for pk, deficit in bucket_deficits.items():
            need_with_buf = max(deficit + 3, int(math.ceil(deficit * 1.45)))
            cands = prompts_by_bucket.get(pk, [])
            delta_prompts.extend(cands[:need_with_buf])
        write_jsonl(PROJECT_ROOT / "data/interim/prompts_delta_to_generate.jsonl", delta_prompts)
    elif not bucket_deficits:
        delta_path = PROJECT_ROOT / "data/interim/prompts_delta_to_generate.jsonl"
        if delta_path.exists():
            delta_path.unlink()

    validate_all_splits(by_split)

    for sp in ("train", "val", "test"):
        rng.shuffle(by_split[sp])
        write_jsonl(args.out_dir / f"{sp}.jsonl", by_split[sp])

    # sample_500.jsonl CHỈ lấy từ Train (400) + Val (100), tuyệt đối không lấy Test (Mục 5.0.2)
    sample_500 = by_split["train"][:400] + by_split["val"][:100]
    write_jsonl(args.out_dir / "sample_500.jsonl", sample_500)

    # Thống kê chi tiết
    stats: Dict[str, Any] = {
        "total_core_samples": sum(len(v) for v in by_split.values()),
        "target_core_samples": 5490,
        "generated_kept": kept_gen,
        "generated_filtered_out": dict(drop_reasons),
        "bucket_deficits_count": sum(bucket_deficits.values()),
        "delta_prompts_written": len(delta_prompts),
        "splits": {},
    }
    for sp in ("train", "val", "test"):
        rows = by_split[sp]
        by_type = collections.Counter(r["doc_type"] for r in rows)
        by_gen = collections.Counter(r["generator"] for r in rows)
        by_dom = collections.Counter(r["domain"] for r in rows)
        avg_words_by_type = {
            dt: round(
                sum(len(r["full_text"].split()) for r in rows if r["doc_type"] == dt)
                / max(1, sum(1 for r in rows if r["doc_type"] == dt)),
                1,
            )
            for dt in ("human", "ai", "mixed_human_ai")
        }
        avg_sents_by_type = {
            dt: round(
                sum(len(r["sentences"]) for r in rows if r["doc_type"] == dt)
                / max(1, sum(1 for r in rows if r["doc_type"] == dt)),
                1,
            )
            for dt in ("human", "ai", "mixed_human_ai")
        }
        stats["splits"][sp] = {
            "total": len(rows),
            "ood_count": sum(1 for r in rows if r["is_ood_generator"]),
            "by_doc_type": dict(by_type),
            "by_generator": dict(by_gen),
            "by_domain": dict(by_dom),
            "avg_words_by_type": avg_words_by_type,
            "avg_sentences_by_type": avg_sents_by_type,
        }

    with open(args.out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("ĐÃ KIỂM ĐỊNH & GHÉP THÀNH CÔNG DỮ LIỆU CORE (v3)!", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
