"""Các hàm tiện ích làm sạch văn bản tiếng Việt, khử artifact AI, tách câu và MinHash dedup."""

from __future__ import annotations

import re
import unicodedata
import zlib
from typing import Dict, List, Optional, Set, Tuple

VIET_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

# Các từ viết tắt phổ biến trong tiếng Việt để không tách nhầm câu tại dấu chấm
ABBREV_PATTERN = re.compile(
    r"\b(?:TP|Tp|TS|PGS|GS|ThS|BS|KS|CN|TT|UBND|HĐND|ĐBQH|TAND|VKSND|TW|T\.Ư|"
    r"Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|v\.v|k\.c|tr|Q|P|TX|TTg)\.$",
    re.IGNORECASE,
)

# Các dòng tiêu đề/chú thích rập khuôn do LLM hoặc trình cào báo/wiki sinh ra
BOILERPLATE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:Tóm tắt|Tác giả|Nguồn|Nguồn bài|Theo|Ảnh|Video|Đồ họa|Bài viết|Tin tức|Xem thêm|Đọc thêm|Từ khóa|Thể loại|"
    r"Tiểu thuyết|Chương\s+\d+|Phần\s+\d+)\s*[:\-].*|"
    r"Bài viết này thuộc danh mục.*|"
    r"==+.*==+|"
    r"(?:Dưới đây là|Sau đây là|Tiếp theo là|Đây là phần viết tiếp).*[:\.]|"
    r"\([Tt]heo\s+[^)]+\)\.?"
    r")\s*$",
    re.IGNORECASE,
)

INLINE_PHOTO_CREDIT_RE = re.compile(
    r"(?:^\s*Ảnh\s*:\s*[^.!?;\n]{1,45}[.!?]\s*|\(\s*[Ảả]nh\s*:\s*[^)]{1,45}\))",
    re.IGNORECASE,
)

END_PHOTO_RE = re.compile(r"\s*Ảnh\s*:\s*[^\n]{1,45}$", re.IGNORECASE)

CHATBOT_PREFIX_RE = re.compile(
    r"^\s*(?:Dưới đây là|Sau đây là|Tiếp theo bài viết|Phần viết tiếp[^:]*:|Chắc chắn rồi[^!.]*[!.])\s*",
    re.IGNORECASE,
)

# Regex loại bỏ mẫu sinh bị lạc sang lời dẫn / link tin liên quan (Mục 3.5)
FORBIDDEN_GEN_RE = re.compile(
    r"(?:Dưới đây là|Phần tiếp theo|Tin liên quan|Xem thêm|Tags|Bài viết liên quan)",
    re.IGNORECASE,
)

# Regex lọc các câu tiêu đề/CTA đặc trưng của Chat LLM (Sailor2/Gemma)
CTA_SENT_RE = re.compile(
    r"(?:Tiêu đề|Mở bài|Thân bài|Kết bài|Kết luận|Tóm lại|Tóm tắt|Lưu ý|Lời khuyên|"
    r"Hãy\s|Đừng bỏ lỡ|Theo dõi\s|Chúc bạn|Hy vọng|Nhìn chung|Tổng kết|---)",
    re.IGNORECASE,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
HTML_URL_RE = re.compile(
    r"(?:<[^>]{2,40}>|https?://\S+|www\.\S+|\balt=|thumb\||\+\s*\+\s*\+)",
    re.IGNORECASE,
)
TERMINAL_PUNCT = (".", "!", "?", "...", '."', ".'", '!"', '?"', ".)", "!)", "?)")


def normalize_text_basic(text: str) -> str:
    """Chuẩn hóa Unicode NFC, gỡ Markdown và quy đồng dấu câu/khoảng trắng (Mục 3.2.3)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\u200b", "")
    # Gỡ markdown bold/italic/heading phổ biến của Gemma-3, Qwen, GPT
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = INLINE_PHOTO_CREDIT_RE.sub(" ", text)

    # Quy đồng ngoặc kép/đơn, gạch ngang, dấu ba chấm (chống shortcut dấu câu giữa báo chí và LLM)
    text = text.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "„": '"',
                "«": '"',
                "»": '"',
                "‘": "'",
                "’": "'",
                "‚": "'",
                "–": "-",
                "—": "-",
                "―": "-",
            }
        )
    )
    text = text.replace("…", "...")
    text = re.sub(r"\.{4,}", "...", text)
    # Đồng nhất các ký tự bị tác giả tập A1 strip khỏi cột Human (/, %, *, #, _, &, +, |, ~, <, >, ^, emoji)
    text = re.sub(r"[/%\*#_&\+\|~<>^]|[\U00010000-\U0010ffff]", "", text)

    # Chuẩn hóa khoảng trắng trước/sau dấu câu và khoảng trắng kép trên từng dòng
    lines_out: List[str] = []
    for line in text.split("\n"):
        line = re.sub(r"\s+([.,!?;:\)\]])", r"\1", line)
        line = re.sub(r"([\(\[])\s+", r"\1", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        lines_out.append(line)
    return "\n".join(lines_out).strip()


def clean_lines_and_edges(
    text: str,
    trim_broken_leading: bool = False,
    trim_broken_trailing: bool = True,
) -> str:
    """Làm sạch dòng rác, cắt câu đầu bị cụt (do cắt prompt 50 tokens) và câu cuối bị ngắt dở."""
    text = normalize_text_basic(text)
    if not text:
        return ""

    text = CHATBOT_PREFIX_RE.sub("", text).strip()

    # Nếu văn bản cắt sau prompt (như AI A1 hoặc Human cắt đầu đối xứng), bỏ đoạn cụt đầu tiên
    if trim_broken_leading:
        m = re.search(r"(?:\.\.\.|[.!?])[\"')\]]*\s+|\n+", text)
        if m:
            text = text[m.end() :].lstrip(" .-:;,\"')\n\t")
        else:
            return ""
    else:
        stripped = text.lstrip()
        if stripped and (stripped[0] in ".,;:!?)]" or stripped[0].islower()):
            m = re.search(r"(?:\.\.\.|[.!?])[\"')\]]*\s+|\n+", stripped)
            if m:
                text = stripped[m.end() :].lstrip(" .-:;,\"')\n\t")

    # Lọc bỏ từng dòng boilerplate (Tác giả:, Nguồn:, Ảnh:, Tóm tắt:, == Wiki ==...)
    kept_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = INLINE_PHOTO_CREDIT_RE.sub(" ", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if BOILERPLATE_LINE_RE.match(line):
            continue
        if not kept_lines and len(line.split()) <= 10 and not line.endswith(TERMINAL_PUNCT):
            continue
        kept_lines.append(line)

    text = "\n".join(kept_lines).strip()
    if not text:
        return ""

    if trim_broken_trailing and not text.endswith(TERMINAL_PUNCT):
        last_pos = -1
        for m in re.finditer(r"(?:\.\.\.|[.!?])[\"')\]]*", text):
            last_pos = m.end()
        if last_pos > 0:
            text = text[:last_pos].strip()
        else:
            return ""

    return text


def clean_generated_continuation(raw_cont: str) -> Optional[str]:
    """Hậu xử lý phần văn bản do Base LLM sinh ra theo Mục 3.5:
    - Loại nếu khớp regex lạc đề/chỉ dẫn.
    - Cắt tại \\n\\n đầu tiên hoặc dòng mới không kết thúc bằng dấu câu.
    - Chuẩn hóa ký tự và bỏ câu cụt cuối.
    """
    if not raw_cont:
        return None
    text = unicodedata.normalize("NFC", raw_cont).replace("\r\n", "\n").replace("\r", "\n").lstrip()
    if not text:
        return None
    if FORBIDDEN_GEN_RE.search(text):
        return None

    # Cắt tại dấu xuống dòng kép đầu tiên (base model hay viết lan sang bài/tiêu đề khác)
    dbl_nl = text.find("\n\n")
    if dbl_nl != -1:
        text = text[:dbl_nl]

    # Chuẩn hóa trước khi kiểm tra dòng không có dấu kết thúc câu
    text = normalize_text_basic(text)
    if not text:
        return None

    lines = [END_PHOTO_RE.sub("", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    kept: List[str] = []
    for i, ln in enumerate(lines):
        is_last = i == len(lines) - 1
        if not is_last and not ln.endswith(TERMINAL_PUNCT):
            # Bỏ qua dòng tiêu đề phụ ngắn ở giữa; giữ nguyên đoạn cuối để trim_broken_trailing cắt câu cụt
            if len(ln.split()) <= 15:
                continue
            break
        kept.append(ln)
    if not kept:
        return None

    return clean_lines_and_edges("\n".join(kept), trim_broken_leading=False, trim_broken_trailing=True)


def length_ok(
    sentences: List[str],
    full_text: str,
    min_sents: int = 6,
    max_sents: int = 24,
    min_words: int = 150,
    max_words: int = 650,
) -> bool:
    """Một hàm lọc độ dài chung cho mọi nhãn (Mục 3.2.5): 6-24 câu và 150-650 từ."""
    n_sents = len(sentences)
    if n_sents < min_sents or n_sents > max_sents:
        return False
    n_words = len(full_text.split())
    if n_words < min_words or n_words > max_words:
        return False
    return True


def is_clean_vietnamese(text: str, min_words: int = 80, max_words: int = 800) -> bool:
    """Kiểm tra văn bản tiếng Việt sạch (loại bỏ tiếng Anh, tiếng Mã Lai, tiếng Trung, HTML/Wiki rác)."""
    if not text:
        return False
    if HTML_URL_RE.search(text):
        return False
    if len(CJK_RE.findall(text)) > 2:
        return False

    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False

    alpha_chars = [c for c in text if c.isalpha()]
    min_alpha = max(50, min_words * 2)
    if len(alpha_chars) < min_alpha:
        return False
    viet_count = sum(1 for c in alpha_chars if c in VIET_DIACRITICS)
    # Văn bản tiếng Việt tự nhiên luôn có tỷ lệ chữ cái mang dấu >= 13% (tiếng Anh/Mã Lai ~ 0%)
    if (viet_count / len(alpha_chars)) < 0.13:
        return False

    return True


def split_sentences(text: str) -> List[str]:
    """Tách câu tiếng Việt bảo toàn từ viết tắt và số thập phân."""
    if not text:
        return []

    sentences: List[str] = []
    for para in text.split("\n"):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue

        start = 0
        for m in re.finditer(r"((?:\.\.\.|[.!?])[\"')\]]*)\s+", para):
            end_punct = m.end(1)
            next_start = m.end()
            prefix = para[start:end_punct]

            if para[m.start(1)] == "." and not para[m.start(1) :].startswith("..."):
                if ABBREV_PATTERN.search(prefix):
                    continue
                if re.search(r"\b[A-ZĐ]\.$", prefix):
                    continue

            if next_start < len(para):
                nxt = para[next_start]
                if not (nxt.isupper() or nxt.isdigit() or nxt in "\"'-("):
                    continue

            sent = prefix.strip()
            if sent:
                sentences.append(sent)
            start = next_start

        tail = para[start:].strip()
        if tail:
            sentences.append(tail)

    merged: List[str] = []
    carry = ""
    for s in sentences:
        if carry:
            s = f"{carry} {s}"
            carry = ""
        if len(s.split()) < 3:
            carry = s
        else:
            merged.append(s)
    if carry and merged:
        merged[-1] = f"{merged[-1]} {carry}"
    elif carry:
        merged.append(carry)

    return [
        s
        for s in merged
        if s.endswith(TERMINAL_PUNCT)
        and not s.endswith("!")
        and not ABBREV_PATTERN.search(s)
        and not BOILERPLATE_LINE_RE.match(s)
        and not CTA_SENT_RE.search(s)
    ]


def build_text_and_spans(sentences: List[str]) -> Tuple[str, List[List[int]]]:
    """Ghép danh sách câu thành full_text chuẩn và trả về sentence_char_spans khớp tuyệt đối."""
    spans: List[List[int]] = []
    parts: List[str] = []
    cursor = 0
    for idx, sent in enumerate(sentences):
        s_clean = re.sub(r"\s+", " ", sent).strip()
        if idx > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(s_clean)
        cursor += len(s_clean)
        spans.append([start, cursor])

    full_text = "".join(parts)
    for (st, ed), s_clean in zip(spans, [re.sub(r"\s+", " ", x).strip() for x in sentences]):
        assert full_text[st:ed] == s_clean, f"Span lệch: {full_text[st:ed]!r} != {s_clean!r}"
    return full_text, spans


def dedup_fingerprint(text: str, n_chars: int = 80) -> str:
    """Tạo khóa lọc nhanh dựa trên n_chars ký tự chữ-số đầu tiên đã chuẩn hóa."""
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w]+", " ", norm).strip()
    return norm[:n_chars]


def _norm_for_minhash(text: str) -> str:
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w\s]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def char_ngrams(text: str, n: int = 5) -> Set[str]:
    """Tạo tập 5-gram ký tự đã chuẩn hóa (lowercase, bỏ dấu câu)."""
    norm = _norm_for_minhash(text)
    if len(norm) <= n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def minhash_signature(text: str, n: int = 5, num_bins: int = 64) -> Tuple[int, ...]:
    """Tính chữ ký One-Permutation MinHash 64-bin cho 5-gram ký tự (tốc độ C via zlib.crc32)."""
    grams = char_ngrams(text, n=n)
    if not grams:
        return (0,) * num_bins
    inf = 1 << 32
    bins = [inf] * num_bins
    crc32 = zlib.crc32
    mask = num_bins - 1
    for g in grams:
        hv = crc32(g.encode("utf-8"))
        b = hv & mask
        val = hv >> 6
        if val < bins[b]:
            bins[b] = val
    # Densify các bin rỗng nếu văn bản quá ngắn
    for i in range(num_bins):
        if bins[i] == inf:
            bins[i] = bins[(i + 1) & mask] if bins[(i + 1) & mask] != inf else 0
    return tuple(bins)


def jaccard_5gram(text_a: str, text_b: str) -> float:
    """Tính Jaccard chính xác trên tập 5-gram ký tự."""
    ga = char_ngrams(text_a, n=5)
    gb = char_ngrams(text_b, n=5)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


class MinHashLSHIndex:
    """Chỉ mục LSH (16 bands x 4 rows) bằng One-Permutation MinHash để lọc nhanh Jaccard >= 0.8."""

    def __init__(self, threshold: float = 0.8, num_bands: int = 16, rows_per_band: int = 4) -> None:
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.num_bins = num_bands * rows_per_band
        self.buckets: List[Dict[Tuple[int, ...], List[int]]] = [{} for _ in range(num_bands)]
        self.docs: List[Tuple[str, Set[int]]] = []

    def _grams_and_sig(self, text: str) -> Tuple[Set[int], Tuple[int, ...]]:
        norm = _norm_for_minhash(text)
        if not norm:
            return set(), (0,) * self.num_bins
        crc32 = zlib.crc32
        num_bins = self.num_bins
        mask = num_bins - 1
        inf = 1 << 32
        bins = [inf] * num_bins
        hashes: Set[int] = set()
        add_hash = hashes.add
        L = len(norm)
        if L <= 5:
            hv = crc32(norm.encode("utf-8"))
            add_hash(hv)
            bins[hv & mask] = hv >> 6
        else:
            b_norm = norm.encode("utf-8")
            # Trượt trên chuỗi ký tự rồi băm utf-8 nhanh
            for i in range(L - 4):
                hv = crc32(norm[i : i + 5].encode("utf-8"))
                add_hash(hv)
                b = hv & mask
                val = hv >> 6
                if val < bins[b]:
                    bins[b] = val
        for i in range(num_bins):
            if bins[i] == inf:
                nxt = bins[(i + 1) & mask]
                bins[i] = nxt if nxt != inf else 0
        return hashes, tuple(bins)

    def query_duplicates(self, text: str) -> List[str]:
        hashes, sig = self._grams_and_sig(text)
        if not hashes:
            return []
        cand_ids: Set[int] = set()
        rpb = self.rows_per_band
        for b_idx in range(self.num_bands):
            band_key = sig[b_idx * rpb : (b_idx + 1) * rpb]
            for doc_idx in self.buckets[b_idx].get(band_key, ()):
                cand_ids.add(doc_idx)
        matches: List[str] = []
        n_q = len(hashes)
        thr = self.threshold
        for doc_idx in cand_ids:
            doc_id, doc_hashes = self.docs[doc_idx]
            inter = len(hashes & doc_hashes)
            union = n_q + len(doc_hashes) - inter
            if union > 0 and (inter / union) >= thr:
                matches.append(doc_id)
        return matches

    def add(self, doc_id: str, text: str) -> None:
        hashes, sig = self._grams_and_sig(text)
        if not hashes:
            return
        doc_idx = len(self.docs)
        self.docs.append((doc_id, hashes))
        rpb = self.rows_per_band
        for b_idx in range(self.num_bands):
            band_key = sig[b_idx * rpb : (b_idx + 1) * rpb]
            self.buckets[b_idx].setdefault(band_key, []).append(doc_idx)
