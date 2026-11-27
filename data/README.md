# Vietnamese AI-Generated & Mixed Human–AI Text Dataset (`data-v1`)

Thư mục này chứa tài liệu mô tả, file mẫu debug ([sample_debug.jsonl](./sample_debug.jsonl)) và bộ dữ liệu chính thức **`data-v1`** tuân thủ [Detailed_Project_Plan_v3.md](../Detailed_Project_Plan_v3.md), được phân phối qua Hugging Face Hub: **`DLProject05/vi-mixed-ai-detection`**.

---

## 1. Cách tải & kiểm tra dữ liệu cho các thành viên

Các thư mục dữ liệu lớn (`processed/`, `calib/`, `stress_test/`, `features_cache/`) lưu trên Hugging Face Hub (không đẩy trực tiếp lên GitHub):

```bash
# 1. Tạo file .env từ mẫu (điền HF_TOKEN nếu repo đang ở chế độ Private)
cp .env.example .env

# 2. Tải toàn bộ dữ liệu (processed, calib, stress_test, features_cache) vào data/
python scripts/download_data.py

# Hoặc chỉ tải các thư mục cần dùng:
python scripts/download_data.py --subdir processed calib stress_test

# 3. Chạy kiểm định QA & Gate G3 (xác nhận schema, zero leakage, MinHash & shortcut probes)
python src/data/validate.py
python src/data/probes.py
```

Cấu trúc thư mục `data/`:
```text
data/
├── README.md                 # Tài liệu mô tả bộ dữ liệu (tracked trên GitHub)
├── sample_debug.jsonl        # 20 mẫu thật chuẩn schema v3 (chỉ từ Train/Val, tracked trên GitHub)
├── processed/                # Tập Core Benchmark (5,486 mẫu: train, val, test, sample_500, dataset_stats.json, probe_report.json)
├── calib/                    # Human Calibration Pool (1,000 mẫu Người: val_calib.jsonl, test_calib.jsonl)
├── stress_test/              # 3 bộ kiểm thử mở rộng (Test-only: VTP, ICCIES, Text-Rewriting)
└── features_cache/           # Đặc trưng xác suất trích từ cặp PhoGPT-4B / Qwen-0.5B (TV2 & TV6)
```

---

## 2. Thống kê Bộ Dữ Liệu Chính Thức (`data-v1`)

### 2.1. Tập Core Benchmark (`data/processed/` — Tổng **5,486 mẫu**)

| Tập | Tổng mẫu | `Human` | `AI thuần` | `Mixed` | In-Domain vs. OOD Generator | Độ dài TB (`Human` / `AI` / `Mixed`) |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`train.jsonl`** | **3,500** | 1,000 | 1,000 | 1,500 | 100% ID (`Sailor2-8B`: 780, `Qwen2.5-7B`: 1,720) | 323.8 / 313.6 / 311.4 từ |
| **`val.jsonl`** | **749** | 220 | 199 | 330 | 100% ID (`Sailor2-8B`: 150, `Qwen2.5-7B`: 379) | 324.4 / 313.3 / 308.5 từ |
| **`test.jsonl`** | **1,237** | 320 | 399 | 518 | **539 ID** (`Sailor2`: 70, `Qwen2.5`: 149, `Human ID`: 170) + **698 OOD Gen** (`Gemma-3-12B`: 200, `Llama-3.1-8B`: 498) + `Human OOD`: 150 | 333.5 / 333.8 / 334.1 từ |
| **Tổng Core** | **5,486** | **1,540** | **1,598** | **2,348** | **0% giao `source_article_id`, MinHash `Jaccard < 0.8`** | **6–24 câu, 150–650 từ** |

* **Phân bố miền trên toàn Core:** Báo chí (`4,705` mẫu — 85.8%), Wikipedia (`486` mẫu — 8.9%), Sách tiếng Việt (`295` mẫu — 5.4%) — cả 3 lớp `Human`, `AI thuần` và `Mixed` đều có đủ 3 miền.
* **`sample_500.jsonl`:** `500` mẫu rút từ `train` (`400`) và `val` (`100`) — **0% mẫu `test`** để bảo đảm kỷ luật tập Test khi debug và chọn siêu tham số.

### 2.2. Tập Human Calibration Pool (`data/calib/` — **1,000 mẫu Người**, `subset = "calib"`)
* **`val_calib.jsonl` (`500` Người):** `400 News A1-val · 60 Wiki · 40 Books` — chỉ dùng để chọn ngưỡng `1% FPR` (không dùng train hay chọn model).
* **`test_calib.jsonl` (`500` Người):** `300 News A1-val-tail · 100 News A1-test · 60 Wiki · 40 Books` — kết hợp với `320` Người của `test.jsonl` thành **820 mẫu Người** để đo `FPR` và `TPR@1%FPR`.

### 2.3. Ba bộ Stress-Test Mở Rộng (`data/stress_test/` — `Test-Only`, `subset = "stress"`)

| File | Số mẫu | Mục đích đánh giá |
| :--- | :---: | :--- |
| **`test_vtp_domain.jsonl`** (ST1 - P0) | **911** | OOD miền văn học Vũ Trọng Phụng (`Human` + `Sailor2-8B` + `Gemma-3-12B`). |
| **`test_iccies_commercial.jsonl`** (ST2 - P0) | **1,000** | LLM thương mại đóng (`500 Human` + `500 AI: GPT-4o-mini, Gemini 1.5 Flash, DeepSeek`), đã khử trùng MinHash với Core & Calib. |
| **`test_rewriting_robustness.jsonl`** (ST3 - P1) | **3,872** | Độ bền qua 5 cấp độ (`human`, `ai_original`, `ai_paraphrase_1`, `ai_paraphrase_2`, `ai_humanized`), đã lọc mẫu lẫn tiếng Mã Lai. |

---

## 3. Kết quả Kiểm định Shortcut (Gate G3 — `data/processed/probe_report.json`)

Huấn luyện Logistic Regression trên `train.jsonl`, đánh giá `AUROC` trên `val.jsonl` (ngưỡng đạt: **`AUROC <= 0.65`**):

| Probe | So sánh | Val AUROC | Kết quả |
| :--- | :--- | :---: | :---: |
| **1. Câu đầu (`first_sentence_edge`)** | `Human` vs `AI thuần` | **0.6353** | **PASS** |
| **2a. Câu cuối (`last_sentence_edge`)** | `Human` vs `AI thuần` | **0.6366** | **PASS** |
| **2b. Câu cuối (`last_sentence_edge`)** | `Human` vs `Mixed` | **0.5596** | **PASS** |
| **3. Độ dài (`length`: từ, câu, từ/câu)** | `Human` vs `AI + Mixed` | **0.5243** | **PASS** |
| **4. Ký tự & Dấu câu (`punctuation`)** | `Human` vs `AI + Mixed` | **0.5914** | **PASS** |
| **5. Cặp câu ranh giới (`boundary_edge`)** | `Human` vs `Mixed` | **0.5116** | **PASS** |

---

## 4. Định dạng Dữ liệu (JSONL Schema v3 — Hợp đồng C1)

```json
{
  "sample_id": "test_mixed_llama_0012",
  "source_article_id": "news_00142",
  "split": "test",
  "subset": "core",
  "is_ood_generator": true,
  "domain": "news",
  "doc_label": 1,
  "doc_type": "mixed_human_ai",
  "generator": "meta-llama/Llama-3.1-8B",
  "gen_config": {"repo": "unsloth/Meta-Llama-3.1-8B", "dtype": "fp16", "temperature": 0.7, "top_p": 0.9},
  "boundary_sentence_idx": 4,
  "full_text": "Câu 1. Câu 2. Câu 3. Câu 4. Câu 5 do AI. Câu 6 do AI.",
  "sentence_char_spans": [[0, 6], [7, 13], [14, 20], [21, 27], [28, 40], [41, 53]],
  "sentences": ["Câu 1.", "Câu 2.", "Câu 3.", "Câu 4.", "Câu 5 do AI.", "Câu 6 do AI."],
  "sentence_labels": [0, 0, 0, 0, 1, 1]
}
```

* `subset`: `"core"`, `"calib"`, hoặc `"stress"`.
* `doc_label`: `0` = 100% Người viết (`human`); `1` = có chứa văn bản AI (`ai` hoặc `mixed_human_ai`).
* `boundary_sentence_idx`: chỉ số `0-based` $b \in \{0,\dots,N\}$ của câu AI đầu tiên (**$b = 0$ nếu 100% AI, $b = N$ nếu 100% Người**; mọi bản ghi đều thỏa bất biến `sentence_labels[i] == int(i >= b)`).
* `sentence_char_spans`: cặp `[start, end]` khớp tuyệt đối `full_text[start:end] == sentences[i]`.
