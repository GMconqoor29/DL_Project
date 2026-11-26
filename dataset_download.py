import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download

# Danh sách nguồn dữ liệu khớp với dataset_sources.txt
DATASETS = {
    # --- Section A: Dataset có cặp Human - AI ---
    "datasets/vietnamese-news": "trieunh/vietnamese.human.ai.news.sets",                          # [A1]
    "datasets/vu-trong-phung": "trieunh/Vietnamese.Ai.Human.Literary.works.VuTrongPhung",         # [A2]
    "datasets/sampled-generations-vtp": "trieunh/SampledGenerations-VTP",                         # [A3]
    "datasets/text-rewriting": "trieunh/Text-Rewriting",                                          # [A4]
    "datasets/iccies-2025-news-human-ai": "AnhNguyen2299/vietnamese_news_human_ai",               # [A5]

    # --- Section B: Dataset chỉ có văn bản người viết (làm mồi sinh mixed / thêm domain) ---
    "datasets/vu-trong-phung-human": "trieunh/Vietnamese_literature_VuTrongPhung",                # [B6]
    "datasets/vietnamese-wikipedia-clean": "lantian520/vietnamese-wikipedia-clean",               # [B8]
    "datasets/vietnamese-book-corpus": "tmnam20/Vietnamese-Book-Corpus",                          # [B9]
    "datasets/binhvq-news-vi": "vietgpt/binhvq_news_vi",                                          # [B7]
}


def download_one(local_dir: str, repo_id: str) -> str:
    print(f"[START] {repo_id} -> {local_dir}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=4,
    )
    print(f"[DONE]  {repo_id} -> {local_dir}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Tải các bộ dữ liệu từ Hugging Face Hub.")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Chỉ tải các thư mục có tên chứa từ khóa này (mặc định: tải tất cả).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Số dataset tải song song cùng lúc (mặc định: 2).",
    )
    args = parser.parse_args()

    items = list(DATASETS.items())
    if args.only:
        items = [
            (d, r)
            for d, r in items
            if any(k.lower() in d.lower() or k.lower() in r.lower() for k in args.only)
        ]

    print(f"Tổng số bộ dữ liệu sẽ kiểm tra/tải: {len(items)}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(download_one, d, r): (d, r) for d, r in items}
        for fut in as_completed(future_map):
            d, r = future_map[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[ERROR] {r} ({d}): {exc}", flush=True)
                failures.append((d, r, str(exc)))

    if failures:
        raise SystemExit(f"Có {len(failures)} bộ dữ liệu lỗi khi tải: {failures}")
    print("Hoàn tất tải toàn bộ dữ liệu!", flush=True)


if __name__ == "__main__":
    main()
