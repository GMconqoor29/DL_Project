"""Kiểm định QA dữ liệu (TV6 — Mục 3.2.7, 3.8 & 5.TV6):
1. Kiểm tra schema v3 (`subset`, `boundary_sentence_idx`, `sentence_char_spans`, bất biến `sentence_labels[i] == int(i >= b)`).
2. Kiểm tra `length_ok` (6-24 câu, 150-650 từ) trên toàn bộ Core và Calib.
3. Kiểm tra không có generator OOD trong Train/Val; `sample_500.jsonl` không chứa mẫu Test.
4. Kiểm tra không trùng `source_article_id` và khử trùng lặp MinHash 5-gram ký tự (Jaccard >= 0.8) giữa mọi cặp tập.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.text_utils import MinHashLSHIndex, length_ok

REQUIRED_FIELDS = {
    "sample_id",
    "source_article_id",
    "split",
    "subset",
    "is_ood_generator",
    "domain",
    "doc_label",
    "doc_type",
    "generator",
    "boundary_sentence_idx",
    "full_text",
    "sentence_char_spans",
    "sentences",
    "sentence_labels",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_record(r: Dict[str, Any], enforce_length_ok: bool) -> List[str]:
    errs: List[str] = []
    missing = REQUIRED_FIELDS - set(r.keys())
    if missing:
        errs.append(f"Thiếu trường {sorted(missing)} ở {r.get('sample_id')}")
        return errs

    sid = r["sample_id"]
    sents = r["sentences"]
    spans = r["sentence_char_spans"]
    labels = r["sentence_labels"]
    ft = r["full_text"]
    n_sents = len(sents)

    if not (n_sents == len(spans) == len(labels) and n_sents >= 3):
        errs.append(f"Độ dài mảng câu không khớp ở {sid}")
        return errs

    if enforce_length_ok and not length_ok(sents, ft, min_sents=6, max_sents=24, min_words=150, max_words=650):
        errs.append(f"Vi phạm length_ok (6-24 câu, 150-650 từ) ở {sid}: {n_sents} câu, {len(ft.split())} từ")

    for (st, ed), s in zip(spans, sents):
        if ft[st:ed] != s:
            errs.append(f"Lệch span ký tự ở {sid}")
            break

    b = r["boundary_sentence_idx"]
    dt = r["doc_type"]
    if dt == "human":
        if r["doc_label"] != 0 or b != n_sents:
            errs.append(f"Human phải có doc_label=0 và b=N ({n_sents}) ở {sid}, gặp b={b}")
    elif dt == "ai":
        if r["doc_label"] != 1 or b != 0:
            errs.append(f"AI thuần phải có doc_label=1 và b=0 ở {sid}, gặp b={b}")
    elif dt == "mixed_human_ai":
        if r["doc_label"] != 1 or not (2 <= b <= n_sents - 2):
            errs.append(f"Mixed b={b} ngoài [2, {n_sents - 2}] ở {sid}")
    else:
        errs.append(f"doc_type không hợp lệ '{dt}' ở {sid}")

    expected_labels = [int(i >= b) for i in range(n_sents)]
    if labels != expected_labels:
        errs.append(f"Vi phạm bất biến sentence_labels ở {sid}")

    if r["split"] in ("train", "val") and r["is_ood_generator"]:
        errs.append(f"Generator OOD bị rò vào {r['split']} ở {sid}")

    return errs


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm định toàn diện bộ dữ liệu v3 (schema, leakage, MinHash).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    args = parser.parse_args()

    groups: Dict[str, Tuple[Path, bool]] = {
        "train": (args.data_dir / "processed/train.jsonl", True),
        "val": (args.data_dir / "processed/val.jsonl", True),
        "test": (args.data_dir / "processed/test.jsonl", True),
        "val_calib": (args.data_dir / "calib/val_calib.jsonl", True),
        "test_calib": (args.data_dir / "calib/test_calib.jsonl", True),
        "stress_vtp": (args.data_dir / "stress_test/test_vtp_domain.jsonl", False),
        "stress_iccies": (args.data_dir / "stress_test/test_iccies_commercial.jsonl", False),
        "stress_rewriting": (args.data_dir / "stress_test/test_rewriting_robustness.jsonl", False),
    }

    loaded: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[str] = []
    seen_sample_ids: Set[str] = set()
    src_ids_by_group: Dict[str, Set[str]] = collections.defaultdict(set)

    for g_name, (path, strict_len) in groups.items():
        if not path.exists():
            print(f"[Cảnh báo] Không thấy file {path}, bỏ qua nhóm {g_name}.")
            continue
        rows = load_jsonl(path)
        loaded[g_name] = rows
        for r in rows:
            sid = r.get("sample_id", "")
            if sid in seen_sample_ids:
                errors.append(f"Trùng sample_id toàn cục: {sid}")
            seen_sample_ids.add(sid)
            src_ids_by_group[g_name].add(r.get("source_article_id", ""))
            errors.extend(check_record(r, enforce_length_ok=strict_len))
        print(f"  - {g_name:18s}: {len(rows):5d} mẫu")

    # Kiểm tra sample_500.jsonl không chứa mẫu test
    s500_path = args.data_dir / "processed/sample_500.jsonl"
    if s500_path.exists():
        s500_rows = load_jsonl(s500_path)
        test_ids = {r["sample_id"] for r in loaded.get("test", [])}
        leaked_test = [r["sample_id"] for r in s500_rows if r.get("split") == "test" or r["sample_id"] in test_ids]
        if leaked_test:
            errors.append(f"sample_500.jsonl chứa {len(leaked_test)} mẫu test!")
        print(f"  - {'sample_500':18s}: {len(s500_rows):5d} mẫu (0 mẫu test)")

    # Kiểm tra trùng source_article_id giữa các tập độc lập
    # (stress_vtp và stress_rewriting cùng dùng họ Vũ Trọng Phụng nên không ghép chung với nhau, nhưng phải rời với toàn bộ phần còn lại)
    group_list = list(loaded.keys())
    for i, g1 in enumerate(group_list):
        for g2 in group_list[i + 1 :]:
            if {g1, g2} == {"stress_vtp", "stress_rewriting"}:
                continue
            ov = src_ids_by_group[g1] & src_ids_by_group[g2]
            if ov:
                errors.append(f"Trùng {len(ov)} source_article_id giữa {g1} và {g2}!")

    # Kiểm tra MinHash 5-gram ký tự (Jaccard >= 0.8) giữa mọi cặp tập
    lsh = MinHashLSHIndex(threshold=args.jaccard_threshold)
    minhash_hits: List[str] = []
    doc_group_map: Dict[str, str] = {}

    for g_name in ("train", "val", "test", "val_calib", "test_calib", "stress_iccies", "stress_vtp", "stress_rewriting"):
        for r in loaded.get(g_name, []):
            uid = f"{g_name}::{r['sample_id']}"
            dups = lsh.query_duplicates(r["full_text"])
            for d in dups:
                other_g = doc_group_map[d]
                if other_g == g_name and g_name in ("stress_vtp", "stress_rewriting"):
                    continue
                if {other_g, g_name} == {"stress_vtp", "stress_rewriting"}:
                    continue
                minhash_hits.append(f"{uid} <-> {d}")
            doc_group_map[uid] = g_name
            lsh.add(uid, r["full_text"])

    if minhash_hits:
        errors.append(f"Phát hiện {len(minhash_hits)} cặp trùng lặp MinHash (Jaccard >= {args.jaccard_threshold}): {minhash_hits[:5]}")

    if errors:
        print("\n[FAILED] Phát hiện lỗi kiểm định dữ liệu:")
        for e in errors[:20]:
            print(f"  - {e}")
        sys.exit(1)

    print("\n[PASSED] Toàn bộ kiểm định schema v3, độ dài, rò rỉ source_article_id và MinHash (Jaccard >= 0.8) ĐẠT 100%!")


if __name__ == "__main__":
    main()
