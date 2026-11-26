# Vietnamese AI-Generated & Mixed Human–AI Text Dataset

Thư mục này chứa tài liệu mô tả, file mẫu debug ([sample_debug.jsonl](./sample_debug.jsonl)) và là nơi lưu trữ cục bộ bộ dữ liệu chuẩn của dự án được host trên Hugging Face Hub: **`DLProject05/vi-mixed-ai-detection`**.

---

## 1. Cách tải dữ liệu

Các tập dữ liệu lớn (`processed/`, `stress_test/`, `features_cache/`) **không** lưu trực tiếp trên GitHub mà được phân phối qua Hugging Face Hub.

```bash
# 1. Tạo file .env từ mẫu (điền HF_TOKEN nếu repo đang ở chế độ Private)
cp .env.example .env

# 2. Tải toàn bộ dữ liệu vào thư mục data/
python scripts/download_data.py

# Hoặc chỉ tải từng thư mục con cụ thể:
python scripts/download_data.py --subdir processed
python scripts/download_data.py --subdir stress_test
```

Cấu trúc thư mục sau khi tải:
```text
data/
├── README.md                 # Tài liệu mô tả bộ dữ liệu (tracked trên GitHub)
├── sample_debug.jsonl        # 20 mẫu thật chuẩn schema (~111 KB, tracked trên GitHub để test nhanh)
├── processed/                # Bộ Core Benchmark (4,996 mẫu: train, val, test, sample_500, dataset_stats.json)
├── stress_test/              # 3 bộ kiểm thử mở rộng (Test-only: Văn học VTP, ICCIES Commercial, Text-Rewriting)
└── features_cache/           # Đặc trưng xác suất trích từ cặp PhoGPT-4B (do TV2 bổ sung ở Tuần 2)
```

---

## 2. Nguồn dữ liệu gốc (Data Sources)

Bộ dữ liệu được xây dựng và chọn lọc từ các bộ ngữ liệu tiếng Việt công khai (chi tiết đầy đủ trong [dataset_sources.txt](../dataset_sources.txt)):

| Ký hiệu | Nguồn Hugging Face | Miền / Đặc điểm | Vai trò trong dự án | License |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | `trieunh/vietnamese.human.ai.news.sets` | Báo chí (`Sailor2-8B` train/val, `Gemma-3-12B` test) | Xương sống cho Tập **Core** (`Train / Val / Core Test`) | Apache-2.0 |
| **B8** | `lantian520/vietnamese-wikipedia-clean` | Wikipedia tiếng Việt (443k trang) | Bổ sung đa dạng miền cho `Human` & mồi `Mixed` | CC-BY-SA |
| **B9** | `tmnam20/Vietnamese-Book-Corpus` | Sách tiếng Việt (16.4k cuốn, lọc đoạn giữa sách sạch) | Bổ sung đa dạng miền cho `Human` & mồi `Mixed` | Public |
| **A2** | `trieunh/Vietnamese.Ai.Human.Literary.works.VuTrongPhung` | Văn học Vũ Trọng Phụng (369 đoạn gốc × 2 LLM) | **Stress-Test 1** – OOD Miền văn học *(100% Test-only)* | Apache-2.0 |
| **A5** | `AnhNguyen2299/vietnamese_news_human_ai` (ICCIES 2025) | Báo chí + LLM thương mại (`GPT-4o-mini`, `Gemini 1.5 Flash`, `DeepSeek`, `Llama 3.3`) | **Stress-Test 2** – Commercial LLMs *(Test-only)* | CC-BY-4.0 |
| **A4** | `trieunh/Text-Rewriting` | Viết lại 4 cấp độ trên 369 đoạn Vũ Trọng Phụng (`Sailor2-8B/14B`, `Gemma-3-12B`) | **Stress-Test 3** – Độ bền Paraphrase & Humanizer *(Test-only)* | MIT |

> **Lưu ý chống rò rỉ dữ liệu:** Do 4 bộ họ *Vũ Trọng Phụng* (`A2`, `A3`, `A4`, `B6`) dùng chung đúng 369 đoạn văn gốc, nhóm **cách ly 100% các bộ này sang tập Stress-Test** (0% nằm trong `Train/Val`).

---

## 3. Quy trình Tiền xử lý & Sinh dữ liệu Mixed (Generation Pipeline)

Toàn bộ quy trình có thể tái lập 100% với `SEED = 42` qua 3 script:
1. [src/data/preprocess.py](../src/data/preprocess.py) — Lọc sạch, tách câu, rút 3 lô rời nhau và tạo 3 bộ Stress-Test.
2. [scripts/kaggle_vllm_generate.py](../scripts/kaggle_vllm_generate.py) / [scripts/kaggle_generate_notebook.ipynb](../scripts/kaggle_generate_notebook.ipynb) — Sinh phần viết tiếp trên Kaggle `GPU T4 x2`.
3. [src/data/merge_generated.py](../src/data/merge_generated.py) — Hậu xử lý phần AI sinh ra, kiểm định và ghép thành `train/val/test.jsonl`.

### Bước 1: Làm sạch & Khử Artifact (`src/data/text_utils.py`)
* **Chuẩn hóa & gỡ định dạng AI:** Chuẩn hóa Unicode NFC, gỡ ký tự Markdown (`**bold**`, `__italic__`, `# Heading`), xóa dòng chú thích ảnh/tiêu đề rác (*"Ảnh: ..."*, *"Tóm tắt:"*, *"Tác giả:"*, *"Nguồn:"*, mã `== Heading ==` của Wiki).
* **Khử lỗi câu cụt của VietBinoculars:** Do dữ liệu AI của `A1/A2/A4` được sinh tiếp sau prompt 50 tokens rồi cắt bỏ prompt, câu đầu tiên thường bị cụt giữa chừng. Pipeline **cắt bỏ phần chữ lẻ đầu bài đến hết dấu chấm câu đầu tiên** và loại bỏ câu cuối chưa hoàn chỉnh.
* **Lọc ngôn ngữ tiếng Việt sạch:** Loại bỏ văn bản chứa HTML/URL, ký tự CJK (`> 2` ký tự), sách tiếng Anh (`B9`) và câu lỗi lẫn tiếng Mã Lai (`A4`) bằng ngưỡng tỷ lệ chữ cái có dấu tiếng Việt $\ge 13\%$.
* **Tách câu & Tọa độ ký tự:** Tách câu bảo toàn các từ viết tắt tiếng Việt (`TP.`, `TS.`, `PGS.`, `Q.`, `P.`, số thập phân), đồng thời sinh mảng `sentence_char_spans` thỏa mãn bất biến `full_text[start:end] == sentences[j]` cho mọi câu.

### Bước 2: Rút 3 Lô bài gốc hoàn toàn rời nhau (`SEED = 42`)
Để tránh trùng lặp bài gốc và không gây nhiễu nhãn ở phần mở đầu, `preprocess.py` chọn lọc các bài sạch (`6–25` câu, `110–620` từ) rồi rút **Stratified Random Sampling** (cân bằng 3 nhóm độ dài `short / medium / long`) thành **3 lô hoàn toàn độc lập (`Disjoint Source Articles`)**:
* **Lô 1 (`Human` — 1,500 mẫu):** Giữ nguyên 100% văn bản Người viết (`A1` + `B8` + `B9`).
* **Lô 2 (`AI thuần` — 1,500 mẫu):** Lấy từ các bài khác Lô 1 gồm `1,000` mẫu `Sailor2-8B` (In-domain), `200` mẫu `Gemma-3-12B` (chỉ ở `Test` OOD), và `300` bài mồi cho `Qwen2.5-7B` sinh mới (cắt bỏ 2 câu mồi sau khi sinh để thành 100% AI).
* **Lô 3 (`Mixed Human–AI` — 2,350 bài mồi):** Rút từ các bài Người hoàn toàn khác Lô 1 & Lô 2; với mỗi bài có $N$ câu ($N \ge 6$), chọn ngẫu nhiên vị trí cắt $k \sim \text{Uniform}(\max(2, \lfloor 0.2N \rfloor), \min(N-2, \lfloor 0.8N \rfloor))$, giữ lại $k$ câu đầu của Người làm prefix.

*(Tập `A1` được khai thác theo đúng 3 file gốc tách biệt: `train.parquet` $\rightarrow$ `Train`; nửa đầu `validation.parquet` $\rightarrow$ `Val`; nửa sau `validation.parquet` $\rightarrow$ `Test In-Domain`; `test.parquet` $\rightarrow$ `Test OOD`.)*

### Bước 3: Sinh văn bản tiếp nối bằng Base LLMs trên Kaggle (`vLLM`)
* Chạy `2,650` prompt (`2,250` cho `Qwen/Qwen2.5-7B` và `400` cho `unsloth/Meta-Llama-3.1-8B` — bản Base mở không khóa quyền) trên Kaggle **`GPU T4 x2`** (`tensor_parallel_size=2`, `dtype=float16`).
* **Lý do chọn Base Model:** Sinh tiếp trực tiếp (raw text completion) vào ngay sau câu thứ $k$, mạch văn tự nhiên và **0% bị dính câu dẫn nhập kiểu chatbot** (*"Dưới đây là phần viết tiếp..."*).
* **Cấu hình sinh:** `temperature = 0.7`, `top_p = 0.9`, `repetition_penalty = 1.08`, và `max_new_tokens` tính tỷ lệ theo độ dài phần đuôi gốc của bài báo để giữ phân phối độ dài cân bằng với lớp `Human`.

### Bước 4: Hậu xử lý & Kiểm định nghiêm ngặt (`src/data/merge_generated.py`)
* Kiểm định tiếng Việt **trực tiếp trên riêng phần AI sinh ra** (không để phần prefix Người che lấp lỗi ngôn ngữ của AI), cắt câu cuối bị ngắt dở, yêu cầu tối thiểu 2 câu AI trọn vẹn (`>= 22` từ) cho `Mixed` và tối thiểu 3 câu (`>= 55` từ) cho `AI thuần`.
* Trong `2,650` mẫu sinh từ Kaggle, giữ lại **2,296 mẫu sạch (86.6%)** và loại bỏ `354` mẫu lỗi/quá ngắn.
* Kiểm chứng `assert` tự động đảm bảo **0% giao nhau `source_article_id`** giữa `train`, `val`, `test` và khớp 100% `sentence_char_spans`.

---

## 4. Thống kê Bộ Dữ Liệu Chính Thức

### 4.1. Tập Core Benchmark (`data/processed/` — Tổng **4,996 mẫu**)

| Tập | Tổng mẫu | `Human` | `AI thuần` | `Mixed` | In-Domain vs. OOD Generator | Độ dài TB (`Human` / `AI` / `Mixed`) |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`train.jsonl`** | **3,270** | 1,000 | 978 | 1,292 | 100% In-Domain (`Sailor2-8B`: 780, `Qwen2.5-7B`: 1,490) | 328.5 / 355.9 / 389.7 từ |
| **`val.jsonl`** | **697** | 220 | 195 | 282 | 100% In-Domain (`Sailor2-8B`: 150, `Qwen2.5-7B`: 327) | 331.2 / 327.3 / 354.3 từ |
| **`test.jsonl`** | **1,029** | 280 | 293 | 456 | **478 In-Domain** + **551 OOD** (`Gemma-3-12B`: 200, `Llama-3.1-8B`: 351) | 328.8 / 386.7 / 383.5 từ |

* **Phân bố miền dữ liệu:** Báo chí (`4,431` mẫu — 88.7%), Wikipedia (`335` mẫu — 6.7%), Sách tiếng Việt (`230` mẫu — 4.6%).
* **Tập rút gọn:** `sample_500.jsonl` (`350 train` / `75 val` / `75 test`) dùng để kiểm thử nhanh pipeline huấn luyện.

### 4.2. Ba bộ Stress-Test Mở Rộng (`data/stress_test/` — `Test-Only`)

| File | Số mẫu | Mục đích đánh giá |
| :--- | :---: | :--- |
| **`test_vtp_domain.jsonl`** | **855** | Đánh giá chuyển miền sang Văn học Vũ Trọng Phụng (`Human` + `Sailor2-8B` + `Gemma-3-12B`). |
| **`test_iccies_commercial.jsonl`** | **1,000** | Đánh giá trên LLM thương mại đóng (`500 Human` + `500 AI: GPT-4o-mini, Gemini 1.5 Flash, DeepSeek`), đã khử trùng 80 ký tự đầu với `Train/Val`. |
| **`test_rewriting_robustness.jsonl`** | **3,819** | Đánh giá độ suy giảm hiệu năng qua 5 cấp độ (`human`, `ai_original`, `ai_paraphrase_1`, `ai_paraphrase_2`, `ai_humanized`), đã lọc sạch mẫu lẫn tiếng Mã Lai. |

---

## 5. Định dạng Dữ liệu (JSONL Schema)

Mỗi dòng trong các file `.jsonl` là một đối tượng JSON theo chuẩn thống nhất:

```json
{
  "sample_id": "train_mixed_qwen_0000",
  "source_article_id": "news_bf3589f91e473f89",
  "split": "train",
  "is_ood_generator": false,
  "domain": "news",
  "doc_label": 1,
  "doc_type": "mixed_human_ai",
  "generator": "Qwen/Qwen2.5-7B",
  "boundary_sentence_idx": 4,
  "full_text": "Câu 1 của người. Câu 2 của người. ... Câu 5 do AI viết tiếp.",
  "sentence_char_spans": [[0, 16], [17, 33], ...],
  "sentences": ["Câu 1 của người.", "Câu 2 của người.", ...],
  "sentence_labels": [0, 0, 0, 0, 1, 1]
}
```

* `doc_label`: `0` = 100% Người viết; `1` = có chứa văn bản AI (`ai` hoặc `mixed_human_ai`).
* `boundary_sentence_idx`: chỉ số `0-based` của câu đầu tiên do AI viết (`-1` nếu `human`, `0` nếu `ai` thuần, `2 <= b <= N - 2` nếu `mixed_human_ai`).
* `sentence_char_spans`: cặp `[start, end]` trên `full_text` cho từng câu, dùng để căn chỉnh chính xác token BPE của `PhoGPT-4B` với từng câu trong nhánh `SeqXGPT`.
