#!/usr/bin/env python3
"""
build_chunks.py — 将 origin_doc/ 中的法律 .docx 文档清洗并切分为
LlamaIndex 可直接加载的 JSON 格式 (TextNode 数组)。

输出: chunks_json/<document_id>.json
格式: [{"id_": "...", "text": "...", "metadata": {...}}, ...]

加载示例:
    from llama_index.core.schema import TextNode
    import json
    with open("chunks_json/xxx.json") as f:
        nodes = [TextNode(**d) for d in json.load(f)]

用法:
  python RAG/build_chunks.py
  python RAG/build_chunks.py --input RAG/origin_doc --output RAG/chunks_json
  python RAG/build_chunks.py --max-chars 900 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from docx import Document as DocxDocument

# ═══════════════════════ Chinese numeral conversion ═══════════════════

_CN_DIGIT = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_num_to_int(s: str) -> int | None:
    """'一百二十三' → 123, '十五' → 15, '一千二百六十' → 1260"""
    if not s:
        return None
    try:
        result = 0
        current = 0
        for ch in s:
            if ch in _CN_DIGIT:
                current = _CN_DIGIT[ch]
            elif ch in _CN_UNIT:
                if current == 0:
                    current = 1
                result += current * _CN_UNIT[ch]
                current = 0
            else:
                return None
        return (result + current) or None
    except Exception:
        return None


# ═══════════════════════ Regex patterns ═══════════════════════════════

_CN = r"[一二三四五六七八九十百千零〇]+"

RE_PART = re.compile(rf"^第({_CN})编")
RE_SUB_PART = re.compile(rf"^第({_CN})分编")
RE_CHAPTER = re.compile(rf"^第({_CN})章")
RE_SECTION = re.compile(rf"^第({_CN})节")
RE_ARTICLE = re.compile(rf"^第({_CN})条")
RE_TOC = re.compile(r"^目\s*录$")
RE_AMEND_ITEM = re.compile(rf"^\s*({_CN})、")


# ═══════════════════════ Text utilities ═══════════════════════════════

def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def collapse_spaces(text: str) -> str:
    """Replace ideographic spaces / multiple whitespace with single space."""
    return re.sub(r"[\u3000\s]+", " ", text).strip()


def parse_filename(filename: str) -> tuple[str, str, str]:
    """Returns (document_id, title, date_iso)."""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    title = parts[0]
    raw = parts[1] if len(parts) == 2 else ""
    if len(raw) == 8 and raw.isdigit():
        date_iso = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    else:
        date_iso = raw
    return stem, title, date_iso


def estimate_tokens(text: str) -> int:
    """Rough token estimate for Chinese text (≈ 0.6 tokens/char)."""
    return max(1, int(len(text) * 0.625))


# ═══════════════════════ Paragraph extraction ═════════════════════════

def extract_paragraphs(docx_path: str) -> list[str]:
    doc = DocxDocument(docx_path)
    return [normalize(p.text) for p in doc.paragraphs]


# ═══════════════════════ TOC removal ══════════════════════════════════

def remove_toc(paragraphs: list[str]) -> tuple[list[str], bool]:
    """Remove table-of-contents block. Returns (cleaned, was_removed)."""
    toc_idx = None
    for i, p in enumerate(paragraphs):
        if RE_TOC.match(collapse_spaces(p)):
            toc_idx = i
            break
    if toc_idx is None:
        return paragraphs, False

    end = toc_idx + 1
    while end < len(paragraphs) and paragraphs[end].strip():
        end += 1
    # include trailing blank line(s)
    while end < len(paragraphs) and not paragraphs[end].strip():
        end += 1

    return paragraphs[:toc_idx] + paragraphs[end:], True


# ═══════════════════════ Document type detection ══════════════════════

def detect_doc_type(paragraphs: list[str]) -> str:
    texts = [collapse_spaces(p) for p in paragraphs if p.strip()]
    article_count = sum(1 for t in texts if RE_ARTICLE.match(t))
    amend_count = sum(1 for t in texts if RE_AMEND_ITEM.match(t))
    if article_count >= 3:
        return "article_law"
    if amend_count >= 3:
        return "amendment"
    return "non_article_text"


# ═══════════════════════ Hierarchy tracker ════════════════════════════

class Hierarchy:
    def __init__(self):
        self.part: str | None = None
        self.sub_part: str | None = None
        self.chapter: str | None = None
        self.section: str | None = None

    def try_update(self, line: str) -> bool:
        """Returns True if the line is a structural heading."""
        c = collapse_spaces(line)
        if RE_PART.match(c):
            self.part = c
            self.sub_part = self.chapter = self.section = None
            return True
        if RE_SUB_PART.match(c):
            self.sub_part = c
            self.chapter = self.section = None
            return True
        if RE_CHAPTER.match(c):
            self.chapter = c
            self.section = None
            return True
        if RE_SECTION.match(c):
            self.section = c
            return True
        return False

    def path(self) -> str:
        return " > ".join(v for v in [self.part, self.sub_part, self.chapter, self.section] if v)

    def to_dict(self) -> dict:
        return {
            "part": self.part,
            "sub_part": self.sub_part,
            "chapter": self.chapter,
            "section": self.section,
        }


# ═══════════════════════ Node builder ═════════════════════════════════

def make_node(
    doc_id: str, index: int, text: str,
    chunk_type: str, chunk_title: str,
    title: str, date: str, filename: str, doc_type: str,
    hierarchy: dict, hierarchy_path: str,
    article_label: str | None = None,
    article_number: int | None = None,
    split_part: int = 1, split_total: int = 1,
) -> dict:
    ccount = len(text)
    return {
        "id_": f"{doc_id}_{index:04d}",
        "text": text,
        "metadata": {
            "document_id": doc_id,
            "title": title,
            "date": date,
            "source_file": filename,
            "document_type": doc_type,
            "chunk_type": chunk_type,
            "chunk_title": chunk_title,
            "hierarchy_path": hierarchy_path,
            **hierarchy,
            "article_label": article_label,
            "article_number": article_number,
            "split_part": split_part,
            "split_total": split_total,
            "char_count": ccount,
            "token_estimate": estimate_tokens(text),
        },
    }


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text exceeding max_chars at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    sentences = re.split(r"(?<=[。；;！!？?])", text)
    buf = ""
    for s in sentences:
        if not s:
            continue
        if buf and len(buf) + len(s) > max_chars:
            pieces.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        pieces.append(buf)

    # fallback: hard split if any piece still too long
    final: list[str] = []
    for p in pieces:
        while len(p) > max_chars:
            final.append(p[:max_chars])
            p = p[max_chars:]
        if p:
            final.append(p)
    return final


# ═══════════════════════ Chunking strategies ══════════════════════════

def _find_content_start(paragraphs: list[str]) -> int:
    """Find where structural/article content begins (skip title+date header)."""
    for i, p in enumerate(paragraphs):
        c = collapse_spaces(p)
        if not c:
            continue
        if (RE_PART.match(c) or RE_SUB_PART.match(c)
                or RE_CHAPTER.match(c) or RE_SECTION.match(c)
                or RE_ARTICLE.match(c)):
            return i
    return len(paragraphs)


def _build_preface(paragraphs: list[str], end: int) -> str:
    lines = [p for p in paragraphs[:end] if p.strip()]
    return "\n".join(lines)


def chunk_article_law(
    paragraphs: list[str], doc_id: str, title: str,
    date: str, filename: str, max_chars: int,
) -> list[dict]:
    nodes: list[dict] = []
    hierarchy = Hierarchy()
    doc_type = "article_law"
    idx = 0  # running node index

    content_start = _find_content_start(paragraphs)
    preface = _build_preface(paragraphs, content_start)
    if preface:
        idx += 1
        nodes.append(make_node(
            doc_id, idx, preface, "preface", "前言",
            title, date, filename, doc_type,
            hierarchy.to_dict(), hierarchy.path(),
        ))

    cur_label: str | None = None
    cur_number: int | None = None
    cur_lines: list[str] = []
    cur_hier: dict = {}
    cur_path: str = ""

    def flush():
        nonlocal idx
        if not cur_lines:
            return
        full = "\n".join(cur_lines)
        parts = split_long_text(full, max_chars)
        for si, seg in enumerate(parts, 1):
            idx += 1
            ct = "article" if cur_label else "text"
            nodes.append(make_node(
                doc_id, idx, seg, ct,
                cur_label or "正文",
                title, date, filename, doc_type,
                cur_hier, cur_path,
                article_label=cur_label,
                article_number=cur_number,
                split_part=si, split_total=len(parts),
            ))

    for i in range(content_start, len(paragraphs)):
        p = paragraphs[i]
        c = collapse_spaces(p)
        if not c:
            continue

        if hierarchy.try_update(c):
            flush()
            cur_lines = []
            cur_label = None
            cur_number = None
            cur_hier = hierarchy.to_dict()
            cur_path = hierarchy.path()
            continue

        m = RE_ARTICLE.match(c)
        if m:
            flush()
            cur_label = f"第{m.group(1)}条"
            cur_number = cn_num_to_int(m.group(1))
            cur_lines = [c]
            cur_hier = hierarchy.to_dict()
            cur_path = hierarchy.path()
            continue

        cur_lines.append(c)

    flush()
    return nodes


def chunk_amendment(
    paragraphs: list[str], doc_id: str, title: str,
    date: str, filename: str, max_chars: int,
) -> list[dict]:
    nodes: list[dict] = []
    doc_type = "amendment"
    idx = 0

    content_start = 0
    for i, p in enumerate(paragraphs):
        if RE_AMEND_ITEM.match(collapse_spaces(p)):
            content_start = i
            break

    preface = _build_preface(paragraphs, content_start)
    if preface:
        idx += 1
        nodes.append(make_node(
            doc_id, idx, preface, "preface", "前言",
            title, date, filename, doc_type,
            {}, "",
        ))

    cur_label: str | None = None
    cur_lines: list[str] = []

    def flush():
        nonlocal idx
        if not cur_lines:
            return
        full = "\n".join(cur_lines)
        parts = split_long_text(full, max_chars)
        for si, seg in enumerate(parts, 1):
            idx += 1
            nodes.append(make_node(
                doc_id, idx, seg, "amendment_item",
                cur_label or "修正项",
                title, date, filename, doc_type,
                {}, "",
                split_part=si, split_total=len(parts),
            ))

    for i in range(content_start, len(paragraphs)):
        p = paragraphs[i]
        c = collapse_spaces(p)
        if not c:
            continue

        m = RE_AMEND_ITEM.match(c)
        if m:
            flush()
            cur_label = f"{m.group(1)}、"
            cur_lines = [c]
            continue

        cur_lines.append(c)

    flush()
    return nodes


def chunk_plain(
    paragraphs: list[str], doc_id: str, title: str,
    date: str, filename: str, max_chars: int,
) -> list[dict]:
    doc_type = "non_article_text"
    lines = [collapse_spaces(p) for p in paragraphs if p.strip()]
    full = "\n".join(lines)
    parts = split_long_text(full, max_chars)

    nodes: list[dict] = []
    for i, seg in enumerate(parts, 1):
        chunk_title = "前言" if i == 1 and len(parts) == 1 else f"段落{i}"
        nodes.append(make_node(
            doc_id, i, seg,
            "preface" if len(parts) == 1 else "text",
            chunk_title,
            title, date, filename, doc_type,
            {}, "",
            split_part=i, split_total=len(parts),
        ))
    return nodes


# ═══════════════════════ Main processing ══════════════════════════════

def process_file(docx_path: str, max_chars: int = 900) -> list[dict]:
    filename = os.path.basename(docx_path)
    doc_id, title, date = parse_filename(filename)

    paragraphs = extract_paragraphs(docx_path)
    paragraphs, _toc_removed = remove_toc(paragraphs)
    doc_type = detect_doc_type(paragraphs)

    if doc_type == "article_law":
        return chunk_article_law(paragraphs, doc_id, title, date, filename, max_chars)
    elif doc_type == "amendment":
        return chunk_amendment(paragraphs, doc_id, title, date, filename, max_chars)
    else:
        return chunk_plain(paragraphs, doc_id, title, date, filename, max_chars)


def main():
    parser = argparse.ArgumentParser(
        description="清洗法律 docx 文档并切分为 LlamaIndex 可读取的 JSON chunks"
    )
    parser.add_argument("--input", default="RAG/origin_doc", help="输入 docx 目录")
    parser.add_argument("--output", default="RAG/chunks_json", help="输出 JSON 目录")
    parser.add_argument("--max-chars", type=int, default=900, help="单 chunk 最大字符数")
    parser.add_argument("--verbose", action="store_true", help="打印每个文件处理详情")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(f for f in input_dir.iterdir() if f.suffix == ".docx")
    if not docx_files:
        print(f"未在 {input_dir} 找到 .docx 文件", file=sys.stderr)
        sys.exit(1)

    total_chunks = 0
    errors: list[str] = []
    type_counts = {"article_law": 0, "amendment": 0, "non_article_text": 0}

    print(f"开始处理 {len(docx_files)} 个文档 ...")

    for fi, fp in enumerate(docx_files, 1):
        try:
            nodes = process_file(str(fp), max_chars=args.max_chars)
            doc_type = nodes[0]["metadata"]["document_type"] if nodes else "unknown"
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            out_path = output_dir / f"{fp.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(nodes, f, ensure_ascii=False, indent=2)

            total_chunks += len(nodes)
            if args.verbose:
                print(f"  [{fi}/{len(docx_files)}] {fp.name} → {len(nodes)} chunks ({doc_type})")
        except Exception as e:
            errors.append(f"{fp.name}: {e}")
            print(f"  [ERROR] {fp.name}: {e}", file=sys.stderr)

    print(f"\n处理完成:")
    print(f"  文档数: {len(docx_files)}")
    print(f"  总 chunks: {total_chunks}")
    print(f"  类型分布: {json.dumps(type_counts, ensure_ascii=False)}")
    print(f"  输出目录: {output_dir}")
    if errors:
        print(f"  错误数: {len(errors)}")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
