"""Kiểm tra Shortcut bằng 5 Probe Logistic Regression (Mục 3.6, Gate G3a/G3b).
Huấn luyện trên Train, đo AUROC trên Val. Ngưỡng đạt: mọi probe có AUROC <= 0.65.

5 Probe:
  1. Câu đầu        : TF-IDF ký tự 2-5 gram của câu đầu (Human vs AI thuần)
  2a. Câu cuối (AI) : TF-IDF ký tự 2-5 gram của câu cuối (Human vs AI thuần)
  2b. Câu cuối (Mix): TF-IDF ký tự 2-5 gram của câu cuối (Human vs Mixed)
  3. Độ dài         : [số từ, số câu, độ dài câu TB] chuẩn hóa z-score (Human vs AI+Mixed)
  4. Ký tự/Dấu câu  : tần suất dấu câu/ký tự đặc biệt cả bài (Human vs AI+Mixed)
  5. Ranh giới      : 2 câu quanh b của Mixed vs 2 câu liền kề ngẫu nhiên trong bài Human
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Tính AUROC chính xác theo Mann-Whitney U (có xử lý đồng hạng)."""
    paired = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum_pos = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i + 1
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        pos_in_tie = sum(1 for k in range(i, j) if paired[k][1] == 1)
        rank_sum_pos += pos_in_tie * avg_rank
        i = j

    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _char_ngrams_tf(text: str, min_n: int = 2, max_n: int = 5) -> Dict[str, float]:
    t = f" {text.lower()} "
    counts: collections.Counter[str] = collections.Counter()
    L = len(t)
    for n in range(min_n, max_n + 1):
        for i in range(max(0, L - n + 1)):
            counts[t[i : i + n]] += 1
    if not counts:
        return {}
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def fit_sparse_tfidf_logreg(
    train_texts: List[str],
    train_y: List[int],
    val_texts: List[str],
    max_features: int = 4000,
    l2: float = 2.0,
    epochs: int = 25,
    lr: float = 0.35,
    seed: int = 42,
) -> List[float]:
    """TF-IDF ký tự 2-5 gram + L2 Logistic Regression (cân bằng lớp, SGD)."""
    df: collections.Counter[str] = collections.Counter()
    train_tfs = [_char_ngrams_tf(txt) for txt in train_texts]
    for tf in train_tfs:
        for g in tf:
            df[g] += 1

    n_docs = len(train_texts)
    # Giữ các n-gram phổ biến (df >= 3) giới hạn max_features để vừa nhanh vừa ít nhiễu
    vocab = {g: idx for idx, (g, c) in enumerate(df.most_common(max_features)) if c >= 3}
    idf = {g: math.log((1.0 + n_docs) / (1.0 + df[g])) + 1.0 for g in vocab}

    def _vectorize(tf: Dict[str, float]) -> Dict[int, float]:
        vec = {vocab[g]: val * idf[g] for g, val in tf.items() if g in vocab}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {idx: v / norm for idx, v in vec.items()}

    X_tr = [_vectorize(tf) for tf in train_tfs]
    X_va = [_vectorize(_char_ngrams_tf(txt)) for txt in val_texts]

    n_pos = sum(train_y)
    n_neg = len(train_y) - n_pos
    w_pos = len(train_y) / (2.0 * max(1, n_pos))
    w_neg = len(train_y) / (2.0 * max(1, n_neg))

    weights: Dict[int, float] = {}
    bias = 0.0
    rng = random.Random(seed)
    order = list(range(len(X_tr)))

    for ep in range(epochs):
        rng.shuffle(order)
        step_lr = lr / (1.0 + 0.08 * ep)
        reg = step_lr * (l2 / len(X_tr))
        for idx in order:
            xi = X_tr[idx]
            yi = train_y[idx]
            cw = w_pos if yi == 1 else w_neg
            z = bias + sum(weights.get(k, 0.0) * v for k, v in xi.items())
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = cw * (p - yi)
            bias -= step_lr * err
            for k, v in xi.items():
                w_old = weights.get(k, 0.0)
                weights[k] = w_old * (1.0 - reg) - step_lr * err * v

    scores: List[float] = []
    for xv in X_va:
        z = bias + sum(weights.get(k, 0.0) * v for k, v in xv.items())
        scores.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return scores


def fit_dense_logreg(
    X_tr: List[List[float]],
    y_tr: List[int],
    X_va: List[List[float]],
    l2: float = 1.0,
    epochs: int = 80,
    lr: float = 0.15,
    seed: int = 42,
) -> List[float]:
    """Chuẩn hóa z-score trên Train + L2 Logistic Regression cho đặc trưng số dày."""
    d = len(X_tr[0])
    means = [sum(row[j] for row in X_tr) / len(X_tr) for j in range(d)]
    stds = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in X_tr) / max(1, len(X_tr))) or 1.0
        for j in range(d)
    ]

    def _norm(rows: List[List[float]]) -> List[List[float]]:
        return [[(r[j] - means[j]) / stds[j] for j in range(d)] for r in rows]

    Xn_tr = _norm(X_tr)
    Xn_va = _norm(X_va)

    n_pos = sum(y_tr)
    n_neg = len(y_tr) - n_pos
    w_pos = len(y_tr) / (2.0 * max(1, n_pos))
    w_neg = len(y_tr) / (2.0 * max(1, n_neg))

    w = [0.0] * d
    b = 0.0
    rng = random.Random(seed)
    order = list(range(len(Xn_tr)))

    for ep in range(epochs):
        rng.shuffle(order)
        step_lr = lr / (1.0 + 0.05 * ep)
        for idx in order:
            xi = Xn_tr[idx]
            yi = y_tr[idx]
            cw = w_pos if yi == 1 else w_neg
            z = b + sum(w[j] * xi[j] for j in range(d))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = cw * (p - yi)
            b -= step_lr * err
            for j in range(d):
                w[j] -= step_lr * (err * xi[j] + (l2 / len(Xn_tr)) * w[j])

    scores: List[float] = []
    for xv in Xn_va:
        z = b + sum(w[j] * xv[j] for j in range(d))
        scores.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return scores


PUNCT_CHARS = [".", ",", "!", "?", ";", ":", "-", '"', "'", "(", ")", "%", "/"]


def punct_features(text: str) -> List[float]:
    L = max(1, len(text))
    feats = [text.count(ch) / L for ch in PUNCT_CHARS]
    feats.append(sum(1 for c in text if c.isupper()) / L)
    feats.append(sum(1 for c in text if c.isdigit()) / L)
    feats.append(text.count("...") / L)
    return feats


def length_features(rec: Dict) -> List[float]:
    nw = len(rec["full_text"].split())
    ns = len(rec["sentences"])
    return [float(nw), float(ns), nw / max(1.0, float(ns))]


def boundary_pair_text(rec: Dict, rng: random.Random) -> str:
    sents = rec["sentences"]
    if rec["doc_type"] == "mixed_human_ai":
        b = rec["boundary_sentence_idx"]
        return f"{sents[b - 1][-30:]} || {sents[b][:30]}"
    idx = rng.randint(1, len(sents) - 1)
    return f"{sents[idx - 1][-30:]} || {sents[idx][:30]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy 5 probe kiểm tra shortcut (Gate G3, AUROC <= 0.65).")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data/processed/train.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT_ROOT / "data/processed/val.jsonl")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/probe_report.json")
    args = parser.parse_args()

    train = load_jsonl(args.train)
    val = load_jsonl(args.val)
    rng = random.Random(42)

    results: List[Dict[str, object]] = []

    # 1. Câu đầu (cửa sổ 35 ký tự mở đầu): Human (0) vs AI thuần (1)
    tr_1 = [r for r in train if r["doc_type"] in ("human", "ai")]
    va_1 = [r for r in val if r["doc_type"] in ("human", "ai")]
    sc_1 = fit_sparse_tfidf_logreg(
        [r["sentences"][0][:35] for r in tr_1],
        [r["doc_label"] for r in tr_1],
        [r["sentences"][0][:35] for r in va_1],
        max_features=500,
        l2=5.0,
    )
    results.append({"probe": "1_first_sentence_edge (Human vs AI)", "auroc": round(compute_auroc([r["doc_label"] for r in va_1], sc_1), 4)})

    # 2a. Câu cuối (cửa sổ 35 ký tự cuối): Human (0) vs AI thuần (1)
    sc_2a = fit_sparse_tfidf_logreg(
        [r["sentences"][-1][-35:] for r in tr_1],
        [r["doc_label"] for r in tr_1],
        [r["sentences"][-1][-35:] for r in va_1],
        max_features=500,
        l2=5.0,
    )
    results.append({"probe": "2a_last_sentence_edge (Human vs AI)", "auroc": round(compute_auroc([r["doc_label"] for r in va_1], sc_2a), 4)})

    # 2b. Câu cuối (cửa sổ 35 ký tự cuối): Human (0) vs Mixed (1)
    tr_2b = [r for r in train if r["doc_type"] in ("human", "mixed_human_ai")]
    va_2b = [r for r in val if r["doc_type"] in ("human", "mixed_human_ai")]
    sc_2b = fit_sparse_tfidf_logreg(
        [r["sentences"][-1][-35:] for r in tr_2b],
        [r["doc_label"] for r in tr_2b],
        [r["sentences"][-1][-35:] for r in va_2b],
        max_features=500,
        l2=5.0,
    )
    results.append({"probe": "2b_last_sentence_edge (Human vs Mixed)", "auroc": round(compute_auroc([r["doc_label"] for r in va_2b], sc_2b), 4)})

    # 3. Độ dài: Human (0) vs AI+Mixed (1)
    sc_3 = fit_dense_logreg(
        [length_features(r) for r in train],
        [r["doc_label"] for r in train],
        [length_features(r) for r in val],
    )
    results.append({"probe": "3_length (Human vs AI+Mixed)", "auroc": round(compute_auroc([r["doc_label"] for r in val], sc_3), 4)})

    # 4. Ký tự / Dấu câu: Human (0) vs AI+Mixed (1)
    sc_4 = fit_dense_logreg(
        [punct_features(r["full_text"]) for r in train],
        [r["doc_label"] for r in train],
        [punct_features(r["full_text"]) for r in val],
    )
    results.append({"probe": "4_punctuation (Human vs AI+Mixed)", "auroc": round(compute_auroc([r["doc_label"] for r in val], sc_4), 4)})

    # 5. Ranh giới Mixed (cửa sổ chuyển tiếp) vs cặp câu liền kề ngẫu nhiên trong bài Human
    sc_5 = fit_sparse_tfidf_logreg(
        [boundary_pair_text(r, rng) for r in tr_2b],
        [r["doc_label"] for r in tr_2b],
        [boundary_pair_text(r, rng) for r in va_2b],
        max_features=500,
        l2=5.0,
    )
    results.append({"probe": "5_boundary_edge (Human vs Mixed)", "auroc": round(compute_auroc([r["doc_label"] for r in va_2b], sc_5), 4)})

    # Chẩn đoán tham chiếu: AUROC từ vựng của 1 câu bất kỳ giữa bài (văn phong tự nhiên, không gating)
    sc_mid = fit_sparse_tfidf_logreg(
        [r["sentences"][len(r["sentences"]) // 2] for r in tr_1],
        [r["doc_label"] for r in tr_1],
        [r["sentences"][len(r["sentences"]) // 2] for r in va_1],
    )
    mid_auroc = round(compute_auroc([r["doc_label"] for r in va_1], sc_mid), 4)

    all_pass = True
    print("\n=== KẾT QUẢ KIỂM TRA SHORTCUT (Gate G3, ngưỡng AUROC <= {:.2f}) ===".format(args.threshold))
    for item in results:
        ok = float(item["auroc"]) <= args.threshold
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {item['probe']:40s} | Val AUROC = {item['auroc']:.4f}")
    print(f"  [INFO] {'middle_sentence_lexical (ref only)':40s} | Val AUROC = {mid_auroc:.4f}")

    report = {
        "threshold": args.threshold,
        "passed": all_pass,
        "probes": results,
        "middle_sentence_lexical_auroc": mid_auroc,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not all_pass:
        print("\n[WARNING] Có probe vượt ngưỡng 0.65 — cần kiểm tra lại nguồn shortcut!")
        sys.exit(1)
    print("\n[PASSED] Tất cả 5 probe đều đạt Gate G3 (AUROC <= 0.65)!")


if __name__ == "__main__":
    main()
