"""Ghép kết quả `generated_continuations.jsonl` (tải từ Kaggle về) với `core_ready_part1.jsonl`
để tạo thành bộ dữ liệu Core chuẩn cuối cùng trong `data/processed/`:
  - `data/processed/train.jsonl`
  - `data/processed/val.jsonl`
  - `data/processed/test.jsonl`
  - `data/processed/sample_500.jsonl` (500 mẫu rút cân bằng cho test nhanh pipeline)
  - `data/processed/dataset_stats.json` (thống kê chi tiết phục vụ viết báo cáo)
  - Cập nhật luôn `data/sample_debug.jsonl` (20 mẫu thật từ cả 3 lớp Human / AI / Mixed)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.text_utils import (
    build_text_and_spans,
    clean_lines_and_edges,
    is_clean_vietnamese,
    split_sentences,
)


def finalize_generated_record(raw_rec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Hậu xử lý phần văn bản do LLM viết tiếp, lọc câu rác/câu cụt cuối và kiểm định tiếng Việt riêng phần AI."""
    prefix_sents: List[str] = raw_rec["prefix_sentences"]
    raw_cont: str = raw_rec.get("raw_continuation", "")

    cleaned_cont = clean_lines_and_edges(
        raw_cont,
        trim_broken_leading=False,
        trim_broken_trailing=True,
    )
    if not cleaned_cont:
        return None, "empty_after_clean"

    ai_sents = split_sentences(cleaned_cont)
    if ai_sents and prefix_sents and ai_sents[0].strip() == prefix_sents[-1].strip():
        ai_sents = ai_sents[1:]

    is_pure_ai = bool(raw_rec.get("drop_prefix_after_gen", False))
    min_ai_sents = 3 if is_pure_ai else 2
    min_ai_words = 55 if is_pure_ai else 22

    if len(ai_sents) < min_ai_sents:
        return None, f"too_few_ai_sents(<{min_ai_sents})"

    # Kiểm định chất lượng tiếng Việt TRỰC TIẾP trên riêng phần AI sinh ra (không để phần prefix Người che lấp!)
    ai_joined = " ".join(ai_sents)
    if not is_clean_vietnamese(ai_joined, min_words=min_ai_words, max_words=800):
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
    if not is_clean_vietnamese(full_text, min_words=55, max_words=950):
        return None, "full_text_length_out_of_range"

    return (
        {
            "sample_id": raw_rec["sample_id"],
            "source_article_id": raw_rec["source_article_id"],
            "split": raw_rec["split"],
            "is_ood_generator": raw_rec["is_ood_generator"],
            "domain": raw_rec["domain"],
            "doc_label": 1,
            "doc_type": doc_type,
            "generator": raw_rec["generator"],
            "boundary_sentence_idx": boundary_idx,
            "full_text": full_text,
            "sentence_char_spans": spans,
            "sentences": final_sents,
            "sentence_labels": sent_labels,
        },
        "ok",
    )


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_all_splits(by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    """Kiểm định nghiêm ngặt schema, span ký tự và chống rò rỉ source_article_id giữa các tập."""
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
            assert len(sents) == len(spans) == len(labels) >= 3, f"Lỗi độ dài mảng câu ở {sid}"
            for (st, ed), s in zip(spans, sents):
                assert ft[st:ed] == s, f"Lệch span ở {sid}"

            if r["doc_type"] == "human":
                assert r["doc_label"] == 0 and r["boundary_sentence_idx"] == -1 and all(x == 0 for x in labels)
            elif r["doc_type"] == "ai":
                assert r["doc_label"] == 1 and r["boundary_sentence_idx"] == 0 and all(x == 1 for x in labels)
            elif r["doc_type"] == "mixed_human_ai":
                b = r["boundary_sentence_idx"]
                assert r["doc_label"] == 1 and 2 <= b <= len(sents) - 2
                assert labels[:b] == [0] * b and labels[b:] == [1] * (len(sents) - b)
            else:
                raise AssertionError(f"doc_type lạ: {r['doc_type']}")

    assert not (src_by_split["train"] & src_by_split["val"]), "Rò rỉ source_id Train <-> Val!"
    assert not (src_by_split["train"] & src_by_split["test"]), "Rò rỉ source_id Train <-> Test!"
    assert not (src_by_split["val"] & src_by_split["test"]), "Rò rỉ source_id Val <-> Test!"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghép dữ liệu Phần 1 và kết quả vLLM thành train/val/test.jsonl.")
    parser.add_argument(
        "--ready",
        type=Path,
        default=PROJECT_ROOT / "data/interim/core_ready_part1.jsonl",
        help="File core_ready_part1.jsonl",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=PROJECT_ROOT / "data/interim/generated_continuations.jsonl",
        help="File generated_continuations.jsonl tải về từ Kaggle",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Thư mục đầu ra data/processed",
    )
    args = parser.parse_args()

    if not args.generated.exists():
        raise SystemExit(f"Chưa tìm thấy file kết quả vLLM tại: {args.generated}")

    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    with open(args.ready, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_split[rec["split"]].append(rec)

    drop_reasons: collections.Counter[str] = collections.Counter()
    kept_gen = 0
    with open(args.generated, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_rec = json.loads(line)
            fin, reason = finalize_generated_record(raw_rec)
            if fin is None:
                drop_reasons[reason] += 1
                continue
            kept_gen += 1
            by_split[fin["split"]].append(fin)

    validate_all_splits(by_split)

    rng = random.Random(42)
    for sp, rows in by_split.items():
        rng.shuffle(rows)
        write_jsonl(args.out_dir / f"{sp}.jsonl", rows)

    # Tạo bộ sample_500.jsonl chuẩn (350 train, 75 val, 75 test)
    sample_500 = by_split["train"][:350] + by_split["val"][:75] + by_split["test"][:75]
    write_jsonl(args.out_dir / "sample_500.jsonl", sample_500)

    # Cập nhật data/sample_debug.jsonl (20 mẫu THẬT 100%: 6 Human, 6 AI thuần, 8 Mixed Qwen & Llama)
    real_debug = (
        [r for r in by_split["train"] if r["doc_type"] == "human"][:6]
        + [r for r in by_split["train"] if r["doc_type"] == "ai"][:4]
        + [r for r in by_split["test"] if r["doc_type"] == "ai" and r["is_ood_generator"]][:2]
        + [r for r in by_split["train"] if r["doc_type"] == "mixed_human_ai"][:5]
        + [r for r in by_split["test"] if r["doc_type"] == "mixed_human_ai" and r["is_ood_generator"]][:3]
    )
    write_jsonl(PROJECT_ROOT / "data/sample_debug.jsonl", real_debug)

    # Xuất file thống kê đầy đủ phục vụ báo cáo khoa học
    stats: Dict[str, Any] = {
        "total_core_samples": sum(len(v) for v in by_split.values()),
        "generated_kept": kept_gen,
        "generated_filtered_out": dict(drop_reasons),
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

    print("ĐÃ KIỂM ĐỊNH & GHÉP THÀNH CÔNG TRỌN BỘ DỮ LIỆU CORE!")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
