"""Các hàm tiện ích làm sạch văn bản tiếng Việt, khử artifact AI và tách câu kèm vị trí ký tự."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

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
    r"(?:Tóm tắt|Tác giả|Nguồn|Nguồn bài|Theo|Ảnh|Video| Đồ họa|Bài viết|Tin tức|Xem thêm|Đọc thêm|Từ khóa|Thể loại|"
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

CHATBOT_PREFIX_RE = re.compile(
    r"^\s*(?:Dưới đây là|Sau đây là|Tiếp theo bài viết|Phần viết tiếp[^:]*:|Chắc chắn rồi[^!.]*[!.])\s*",
    re.IGNORECASE,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
HTML_URL_RE = re.compile(
    r"(?:<[^>]{2,40}>|https?://\S+|www\.\S+|\balt=|thumb\||\+\s*\+\s*\+)",
    re.IGNORECASE,
)
TERMINAL_PUNCT = (".", "!", "?", "…", '."', ".”", ".'", "!”", "?”", ".)", "!)", "?)")


def normalize_text_basic(text: str) -> str:
    """Chuẩn hóa Unicode NFC, gỡ ký tự Markdown (**bold**, ###) và khoảng trắng thừa."""
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
    return text.strip()


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

    # Nếu văn bản là phần sinh tiếp đã cắt prompt (như VietBinoculars AI), luôn bỏ đoạn cụt đầu tiên
    if trim_broken_leading:
        m = re.search(r"[.!?…][\"”')]*\s+|\n+", text)
        if m:
            text = text[m.end() :].lstrip(" .-–—:;,\"”')\n\t")
        else:
            return ""
    else:
        stripped = text.lstrip()
        if stripped and (stripped[0] in ".,;:!?…)]" or stripped[0].islower()):
            m = re.search(r"[.!?…][\"”')]*\s+|\n+", stripped)
            if m:
                text = stripped[m.end() :].lstrip(" .-–—:;,\"”')\n\t")

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
        for m in re.finditer(r"[.!?…][\"”')]*", text):
            last_pos = m.end()
        if last_pos > 0:
            text = text[:last_pos].strip()
        else:
            return ""

    return text


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
        for m in re.finditer(r"([.!?…][\"”')]*)\s+", para):
            end_punct = m.end(1)
            next_start = m.end()
            prefix = para[start:end_punct]

            if para[m.start(1)] == ".":
                if ABBREV_PATTERN.search(prefix):
                    continue
                if re.search(r"\b[A-ZĐ]\.$", prefix):
                    continue

            if next_start < len(para):
                nxt = para[next_start]
                if not (nxt.isupper() or nxt.isdigit() or nxt in "\"“'‘-–—("):
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

    return [s for s in merged if not BOILERPLATE_LINE_RE.match(s)]


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
    """Tạo khóa khử trùng lặp dựa trên n_chars ký tự chữ-số đầu tiên đã chuẩn hóa."""
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w]+", " ", norm).strip()
    return norm[:n_chars]
