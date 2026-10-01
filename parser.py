"""
parser.py - Architecture document parser for the CodeAgent_DeepSeek pipeline.

Reads the two source documents that describe the target system:

    * ``Architecture_Documentation.md`` - prose description of the architecture
    * ``Architecture_View.md``          - PlantUML-based architecture views

Both documents are cleaned (line endings, trailing whitespace, blank-run
collapsing, PlantUML wrapper/comment noise) and structured into a single
JSON payload that ``agent_core.py`` sends to the DeepSeek API.

Standalone usage::

    python parser.py
    python parser.py --doc Architecture_Documentation.md ^
                     --view Architecture_View.md ^
                     --out architecture_payload.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "DOC_FILE_NAME",
    "VIEW_FILE_NAME",
    "DEFAULT_PROJECT_NAME",
    "normalize_line_endings",
    "read_text_file",
    "clean_text",
    "clean_plantuml",
    "extract_plantuml_blocks",
    "extract_sections",
    "build_document_payload",
    "build_payload",
    "save_payload",
    "main",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DOC_FILE_NAME = "Architecture_Documentation.md"
VIEW_FILE_NAME = "Architecture_View.md"
DEFAULT_PROJECT_NAME = "Space Fractions"
PAYLOAD_SCHEMA_VERSION = "1.0"

_PLANTUML_BLOCK_RE = re.compile(
    r"```(?:plantuml|puml|uml)[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")
_STARTUML_RE = re.compile(r"^[ \t]*@startuml\b", re.IGNORECASE)
_ENDUML_RE = re.compile(r"^[ \t]*@enduml\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"^[ \t]*```")


# --------------------------------------------------------------------------- #
# Reading and cleaning
# --------------------------------------------------------------------------- #

def normalize_line_endings(text: str) -> str:
    """Convert CRLF / CR line endings to plain LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_file(path: Path) -> str:
    """Read a UTF-8 markdown file, tolerating an optional BOM."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8-sig", errors="replace")


def clean_text(text: str) -> str:
    """Clean markdown prose without destroying its structure.

    * normalises line endings to LF,
    * strips trailing whitespace on every line,
    * collapses runs of blank lines into a single blank line,
    * strips leading/trailing blank lines of the document.
    """
    lines = normalize_line_endings(text).split("\n")
    cleaned: List[str] = []
    blank_run = 0
    for line in lines:
        line = line.rstrip()
        if line:
            blank_run = 0
            cleaned.append(line)
            continue
        blank_run += 1
        if blank_run <= 1:
            cleaned.append("")
    return "\n".join(cleaned).strip("\n")


def clean_plantuml(code: str, drop_comments: bool = True) -> str:
    """Clean a PlantUML diagram body.

    * removes ``@startuml`` / ``@enduml`` wrapper lines,
    * removes full-line ``'`` comments when ``drop_comments`` is true,
    * de-indents the block and collapses blank runs.
    """
    lines: List[str] = []
    for raw in normalize_line_endings(code).split("\n"):
        line = raw.rstrip()
        if _STARTUML_RE.match(line) or _ENDUML_RE.match(line):
            continue
        if drop_comments and line.lstrip().startswith("'"):
            continue
        lines.append(line)

    body = textwrap.dedent("\n".join(lines)).strip("\n")

    cleaned: List[str] = []
    blank_run = 0
    for line in body.split("\n"):
        line = line.rstrip()
        if line:
            blank_run = 0
            cleaned.append(line)
            continue
        blank_run += 1
        if blank_run <= 1:
            cleaned.append("")
    return "\n".join(cleaned).strip("\n")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_plantuml_blocks(text: str) -> List[Dict[str, Any]]:
    """Extract every fenced PlantUML block, cleaned and indexed."""
    blocks: List[Dict[str, Any]] = []
    normalized = normalize_line_endings(text)
    for index, match in enumerate(_PLANTUML_BLOCK_RE.finditer(normalized), start=1):
        blocks.append(
            {
                "index": index,
                "language": "plantuml",
                "content": clean_plantuml(match.group(1)),
            }
        )
    return blocks


def extract_sections(text: str) -> Dict[str, Any]:
    """Split cleaned markdown into a flat, ordered list of heading sections.

    Returns ``{"title": <first H1 or "">, "sections": [...]}`` where each
    section is ``{"level": int, "title": str, "content": str}``.  Headings
    inside fenced code blocks are ignored.
    """
    lines = normalize_line_endings(text).split("\n")
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    buffer: List[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buffer, current
        content = "\n".join(buffer).strip("\n")
        if current is None:
            if content.strip():
                sections.append({"level": 0, "title": "preamble", "content": content})
        else:
            current["content"] = content
            sections.append(current)
        buffer = []
        current = None

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1 and title is None and heading:
                title = heading
            current = {"level": level, "title": heading}
            continue

        buffer.append(line)

    flush()

    if title is None and sections:
        title = sections[0]["title"] or ""

    return {"title": title or "", "sections": sections}


# --------------------------------------------------------------------------- #
# Payload assembly
# --------------------------------------------------------------------------- #

def build_document_payload(source_name: str, raw_text: str) -> Dict[str, Any]:
    """Build the cleaned, structured representation of one source document."""
    cleaned = clean_text(raw_text)
    parsed = extract_sections(cleaned)
    diagrams = extract_plantuml_blocks(cleaned)
    return {
        "source": source_name,
        "title": parsed["title"],
        "characters": len(cleaned),
        "section_count": len(parsed["sections"]),
        "plantuml_diagram_count": len(diagrams),
        "sections": parsed["sections"],
        "plantuml_diagrams": diagrams,
    }


def build_payload(
    documentation_path: Path,
    architecture_view_path: Path,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    """Read both architecture documents and build the final JSON payload."""
    documentation_path = Path(documentation_path)
    architecture_view_path = Path(architecture_view_path)

    documentation = build_document_payload(
        documentation_path.name, read_text_file(documentation_path)
    )
    architecture_view = build_document_payload(
        architecture_view_path.name, read_text_file(architecture_view_path)
    )

    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": project_name,
        "documents": {
            "documentation": documentation,
            "architecture_view": architecture_view,
        },
    }


def save_payload(payload: Dict[str, Any], output_path: Path) -> Path:
    """Write the payload as pretty UTF-8 JSON and return the resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Parse the architecture documents into a clean JSON payload."
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=script_dir / DOC_FILE_NAME,
        help=f"Path to the documentation markdown (default: {DOC_FILE_NAME} next to this script)",
    )
    parser.add_argument(
        "--view",
        type=Path,
        default=script_dir / VIEW_FILE_NAME,
        help=f"Path to the architecture view markdown (default: {VIEW_FILE_NAME} next to this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "architecture_payload.json",
        help="Where to write the JSON payload",
    )
    parser.add_argument(
        "--project-name",
        default=DEFAULT_PROJECT_NAME,
        help="Project name recorded in the payload",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_payload(args.doc, args.view, project_name=args.project_name)
    except FileNotFoundError as exc:
        print(f"[parser] ERROR: {exc}", file=sys.stderr)
        return 2

    out_path = save_payload(payload, args.out)
    docs = payload["documents"]
    print(
        f"[parser] documentation : {docs['documentation']['source']} "
        f"({docs['documentation']['section_count']} sections, "
        f"{docs['documentation']['plantuml_diagram_count']} diagrams)"
    )
    print(
        f"[parser] architecture  : {docs['architecture_view']['source']} "
        f"({docs['architecture_view']['section_count']} sections, "
        f"{docs['architecture_view']['plantuml_diagram_count']} diagrams)"
    )
    print(f"[parser] payload written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
