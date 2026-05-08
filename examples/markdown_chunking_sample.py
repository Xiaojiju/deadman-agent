"""Markdown 切块示例：演示「结构优先 → 原子块不切断 → 超长再切 + 元数据」逻辑。

运行：在项目根目录执行
    uv run python examples/markdown_chunking_sample.py

说明：为便于阅读，使用行扫描 + 少量正则，非完整 CommonMark 解析器；
生产环境可换成 markdown-it / mistune 等 AST，再把本脚本的 Phase 2 映射到 TextChunk 即可。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils.file.file_loader import ChunkMetaKey, ChunkType, FileLoadResult, TextChunk

_RAW_TO_CHUNK: dict[
    Literal["paragraph", "code_fence", "table", "front_matter"],
    ChunkType,
] = {
    "paragraph": "md_section",
    "code_fence": "md_code_fence",
    "table": "md_table",
    "front_matter": "md_front_matter",
}

# 示例用字符上限；生产可换成 tokenizer 计数
MAX_EMBED_CHARS = 380


def _github_style_slug(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", t, flags=re.UNICODE)
    t = re.sub(r"[\s_-]+", "-", t).strip("-")
    return t or "section"


def _heading_match(line: str) -> tuple[int, str] | None:
    m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def _looks_like_table_row(line: str) -> bool:
    s = line.strip()
    return "|" in s and s.count("|") >= 2


@dataclass
class _RawBlock:
    """Phase 1：从原文切出的结构单元（尚未做长度合并/再切）。"""

    kind: Literal["paragraph", "code_fence", "table", "front_matter"]
    text: str
    start_line: int
    end_line: int
    heading_titles: tuple[str, ...]
    heading_level: int


def _heading_path_str(titles: tuple[str, ...]) -> str:
    return " > ".join(titles) if titles else ""


def parse_markdown_to_raw_blocks(source: str) -> list[_RawBlock]:
    """Phase 1：按标题栈更新语义边界，代码块/表格整块，其余为段落。"""
    lines = source.splitlines()
    n = len(lines)
    i = 0
    blocks: list[_RawBlock] = []
    stack: list[tuple[int, str]] = []

    def current_heading_meta() -> tuple[tuple[str, ...], int]:
        titles = tuple(t for _, t in stack)
        level = stack[-1][0] if stack else 0
        return titles, level

    # 可选 YAML front matter
    if n >= 2 and lines[0].strip() == "---":
        j = 1
        while j < n and lines[j].strip() != "---":
            j += 1
        if j < n and lines[j].strip() == "---":
            fm_text = "\n".join(lines[1:j])
            blocks.append(
                _RawBlock(
                    kind="front_matter",
                    text=fm_text,
                    start_line=1,
                    end_line=j + 1,
                    heading_titles=(),
                    heading_level=0,
                )
            )
            i = j + 1

    while i < n:
        raw = lines[i]
        line_no = i + 1
        stripped = raw.strip()

        if stripped == "":
            i += 1
            continue

        hm = _heading_match(raw)
        if hm:
            level, title = hm
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            i += 1
            continue

        titles, hlevel = current_heading_meta()

        if stripped.startswith("```"):
            fence = stripped[:3]
            body: list[str] = []
            start = line_no
            i += 1
            while i < n:
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                body.append(lines[i])
                i += 1
            end = i
            blocks.append(
                _RawBlock(
                    kind="code_fence",
                    text="\n".join(body),
                    start_line=start,
                    end_line=end,
                    heading_titles=titles,
                    heading_level=hlevel,
                )
            )
            continue

        if _looks_like_table_row(raw):
            tbl: list[str] = []
            start = line_no
            while i < n and _looks_like_table_row(lines[i]):
                tbl.append(lines[i])
                i += 1
            end = i
            blocks.append(
                _RawBlock(
                    kind="table",
                    text="\n".join(tbl),
                    start_line=start,
                    end_line=end,
                    heading_titles=titles,
                    heading_level=hlevel,
                )
            )
            continue

        para: list[str] = []
        start = line_no
        while i < n:
            ln = lines[i]
            if ln.strip() == "":
                break
            if _heading_match(ln) or ln.strip().startswith("```") or _looks_like_table_row(ln):
                break
            para.append(ln)
            i += 1
        end = i
        text = "\n".join(para).strip()
        if text:
            blocks.append(
                _RawBlock(
                    kind="paragraph",
                    text=text,
                    start_line=start,
                    end_line=end,
                    heading_titles=titles,
                    heading_level=hlevel,
                )
            )
        else:
            i += 1

    return blocks


def raw_blocks_to_text_chunks(
    raw_blocks: list[_RawBlock],
    *,
    source_path: str,
    max_chars: int = MAX_EMBED_CHARS,
) -> tuple[TextChunk, ...]:
    """Phase 2：映射 chunk_type + 元数据；超长块按字符窗口拆分并带 part_*。"""
    chunks: list[TextChunk] = []
    idx = 0

    for rb in raw_blocks:
        ctype = _RAW_TO_CHUNK[rb.kind]

        path = _heading_path_str(rb.heading_titles)
        last_title = rb.heading_titles[-1] if rb.heading_titles else ""
        anchor = _github_style_slug(last_title) if last_title else ""

        def embed_text(body: str) -> str:
            if path:
                return f"[{_heading_path_str(rb.heading_titles)}]\n{body}"
            return body

        body = rb.text
        if len(body) <= max_chars:
            chunks.append(
                TextChunk(
                    text=embed_text(body),
                    chunk_type=ctype,
                    index=idx,
                    metadata={
                        ChunkMetaKey.SOURCE_PATH: source_path,
                        ChunkMetaKey.HEADING_PATH: path,
                        ChunkMetaKey.HEADING_LEVEL: rb.heading_level,
                        ChunkMetaKey.ANCHOR: anchor,
                        ChunkMetaKey.START_LINE: rb.start_line,
                        ChunkMetaKey.END_LINE: rb.end_line,
                        ChunkMetaKey.MD_KIND: rb.kind,
                    },
                )
            )
            idx += 1
            continue

        # 超长：同一结构块拆多段向量，便于回溯同一小节
        total_parts = (len(body) + max_chars - 1) // max_chars
        for p in range(total_parts):
            slice_ = body[p * max_chars : (p + 1) * max_chars]
            chunks.append(
                TextChunk(
                    text=embed_text(slice_),
                    chunk_type=ctype,
                    index=idx,
                    metadata={
                        ChunkMetaKey.SOURCE_PATH: source_path,
                        ChunkMetaKey.HEADING_PATH: path,
                        ChunkMetaKey.HEADING_LEVEL: rb.heading_level,
                        ChunkMetaKey.ANCHOR: anchor,
                        ChunkMetaKey.START_LINE: rb.start_line,
                        ChunkMetaKey.END_LINE: rb.end_line,
                        ChunkMetaKey.MD_KIND: rb.kind,
                        ChunkMetaKey.PART_INDEX: p,
                        ChunkMetaKey.PART_COUNT: total_parts,
                    },
                )
            )
            idx += 1

    return tuple(chunks)


def markdown_to_file_load_result(md: str, source_path: str = "demo.md") -> FileLoadResult:
    raw = parse_markdown_to_raw_blocks(md)
    parts = raw_blocks_to_text_chunks(raw, source_path=source_path)
    return FileLoadResult(
        source_path=source_path,
        format="markdown",
        mime_type="text/markdown",
        text=md,
        chunks=parts,
        structured={"raw_block_count": len(raw)},
        metadata={},
    )


SAMPLE = """---
title: Demo
tags: [rag]
---

# 指南

短介绍一句。

## 安装

先装依赖。

```bash
pip install -e .
echo done
```

## 配置

| key | value |
|-----|-------|
| a   | 1     |

### 高级

这里有一段故意写得很长以便触发按字符再切分：""" + ("字" * 500) + """

## 附录

收尾。
"""


def main() -> None:
    res = markdown_to_file_load_result(SAMPLE, source_path="examples/demo.md")
    print(f"共 {len(res.chunks)} 个 TextChunk\n")
    for c in res.chunks:
        preview = c.text[:80].replace("\n", "\\n")
        if len(c.text) > 80:
            preview += "..."
        print(f"[{c.index}] type={c.chunk_type}")
        print(f"    text: {preview}")
        print(f"    meta: {dict(c.metadata)}")
        print()


if __name__ == "__main__":
    main()
