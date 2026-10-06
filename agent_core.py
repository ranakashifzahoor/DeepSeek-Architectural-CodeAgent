"""
agent_core.py - DeepSeek generation engine for the CodeAgent_DeepSeek pipeline.

Takes the structured architecture payload produced by ``parser.py`` and asks
the official DeepSeek API (OpenAI-compatible ``/v1/chat/completions``) to
generate the complete, production-ready "Space Fractions" project - a
standalone desktop application:

    Game (desktop GUI):
        main_game.py        Tkinter game window (dark space theme, score,
                            streak, level, answer checking) - no web server
    Support files:
        requirements.txt, schema.sql, openapi.yaml, Dockerfile, README.md
    Engine traceability:
        architecture_payload.json

The generated game runs with ``python main_game.py``: a single-file Tkinter
application that picks a random fraction challenge, computes the expected
decimal dynamically, and validates answers (trimmed input, decimal strings
like "0.75", fraction answers like "3/4" compared numerically with a small
tolerance).

The generation is scoped to this essential file set so the output stays
compact.  Nested paths coming back from the model are flattened to their
core file names, and anything outside the core set is skipped (and logged).

Robustness: if an API response is cut off mid-file (token truncation), the
engine does NOT crash.  It salvages what it can by closing the open file
streams, and missing core files are completed with high-quality static
versions - so the run always produces a fully functional project (a playable
game with accurate validation) even in the worst case.

Entry points (both supported)::

    generate_project(payload, output_dir, api_key=api_key, log=...)
    generate_project_code(api_key=KEY, parsed_architecture=..., output_dir=...)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

__all__ = [
    "DEEPSEEK_API_URL",
    "DEFAULT_MODEL",
    "DEFAULT_STAGES",
    "CORE_PROJECT_FILES",
    "DeepSeekClient",
    "parse_generated_files",
    "write_generated_files",
    "build_system_prompt",
    "build_stage_messages",
    "generate_project",
    "generate_project_code",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 600  # seconds - full project stages can take a while
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
ENV_API_KEY = "DEEPSEEK_API_KEY"

# The essential core file set (model-generated).  The engine additionally
# writes architecture_payload.json, so a complete run yields 7 files total.
CORE_PROJECT_FILES: Sequence[str] = (
    "main_game.py",
    "requirements.txt",
    "schema.sql",
    "openapi.yaml",
    "Dockerfile",
    "README.md",
)

_CORE_LOOKUP: Dict[str, str] = {name.lower(): name for name in CORE_PROJECT_FILES}
_CORE_LOOKUP.update(
    {
        "openapi.yml": "openapi.yaml",
        "readme": "README.md",
        "main.py": "main_game.py",
        "game.py": "main_game.py",
    }
)

DEFAULT_STAGES: Sequence[Dict[str, str]] = (
    {
        "name": "app",
        "description": "Standalone Desktop Game (Python + Tkinter GUI)",
        "instruction": (
            "Generate main_game.py and requirements.txt. main_game.py must "
            "be a complete, single-file Tkinter desktop application using "
            "only the Python standard library. It must open a dark "
            "space-themed window (dark background, light text, one warm "
            "accent color) with a large current-fraction display, score, "
            "streak and level counters, an answer entry field, a Check "
            "Answer button, a Next Challenge button and a status message "
            "line. On startup and on every Next Challenge click it must "
            "pick a new random fraction from a pool of at least eight (for "
            "example 1/2, 3/4, 2/5, 5/8, 1/4, 4/5, 3/10, 9/10), compute "
            "the expected decimal dynamically from numerator / denominator "
            "at that moment (do not hardcode it), never repeat the same "
            "fraction twice in a row, advance the level counter, and clear "
            "the answer field so no placeholder such as -- ever remains. "
            "Checking an answer must trim the input, accept decimals like "
            "0.75 (including 0.750) and fractions like 3/4 (parsed as "
            "numerator / denominator and compared numerically with a small "
            "tolerance of about 0.000001), add 100 to the score and grow "
            "the streak on success, reset the streak and show the expected "
            "decimal on a wrong answer, and always leave the input cleared "
            "and both buttons enabled. requirements.txt must state that "
            "the game needs only the Python standard library (tkinter), "
            "with no third-party packages."
        ),
    },
    {
        "name": "database",
        "description": "SQL Schemas & Database Setup",
        "instruction": (
            "Generate schema.sql (full SQL DDL with tables, primary/foreign "
            "keys and constraints) in the project root."
        ),
    },
    {
        "name": "docs",
        "description": "Documentation & Interface Specification",
        "instruction": (
            "Generate openapi.yaml (OpenAPI 3.0 documenting the game's "
            "level and validate operations as its documented interface) "
            "and README.md in the project root. The README must include "
            "the run instructions: python main_game.py, plus a short "
            "feature and gameplay overview."
        ),
    },
    {
        "name": "container",
        "description": "Container Packaging",
        "instruction": (
            "Generate the Dockerfile: a slim Python base image that copies "
            "the project and runs the game (default command: python "
            "main_game.py), with a brief comment that a display is needed "
            "for the GUI."
        ),
    },
)

_REQUIRED_FILES = """\
- main_game.py: the complete standalone Tkinter desktop game (dark space
  themed window with score, streak, level, fraction display, answer entry,
  Check Answer and Next Challenge buttons) with robust answer validation
  (trimmed input, decimals like 0.75, fractions like 3/4, small float
  tolerance) - standard library only, no web server or HTTP code.
- requirements.txt: dependency notes for the game (Python standard library
  only, no third-party packages).
- schema.sql: full SQL DDL (tables, primary/foreign keys, constraints).
- openapi.yaml: OpenAPI 3.0 document describing the game level and
  validate operations.
- Dockerfile: slim Python image that runs the game (python main_game.py).
- README.md: run instructions (python main_game.py) and a gameplay overview.
- No web files: do not emit server.js, package.json, public/index.html,
  Express, Flask or any other web code - the game is a desktop app.
"""

_FILE_BLOCK_RE = re.compile(
    r"^===FILE:[ \t]*(?P<path>.+?)[ \t]*===[ \t]*$\n(?P<content>.*?)^===END FILE===[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_FILE_START_RE = re.compile(r"^===FILE:", re.MULTILINE)
_FILE_END_RE = re.compile(r"^===END FILE===", re.MULTILINE)

_STATIC_FALLBACK_ORDER: Sequence[str] = (
    "main_game.py",
    "requirements.txt",
    "schema.sql",
    "openapi.yaml",
    "Dockerfile",
    "README.md",
    "architecture_payload.json",
)

_STATIC_FILES: Dict[str, str] = {
    "main_game.py": '''\
"""Space Fractions - standalone desktop game.

Run with:  python main_game.py

A dark, space-themed game: convert the fraction shown in the observatory
into a decimal.  Correct answers build score and streak; a wrong answer
shows the expected decimal and resets the streak.
"""

import random
import tkinter as tk

# ---------------------------------------------------------------------------
# Palette (dark space theme)
# ---------------------------------------------------------------------------
BG = "#0B1020"
PANEL = "#101A35"
LINE = "#26304D"
INK = "#E9EDF6"
MUTED = "#93A1BC"
ACCENT = "#F2B94B"
ACCENT_DIM = "#D9A23A"
OK = "#5FBF8F"
BAD = "#E07777"

TOLERANCE = 0.000001

CHALLENGES = (
    ("1/2", 1, 2),
    ("3/4", 3, 4),
    ("2/5", 2, 5),
    ("5/8", 5, 8),
    ("1/4", 1, 4),
    ("4/5", 4, 5),
    ("3/10", 3, 10),
    ("9/10", 9, 10),
)


def parse_answer(text):
    """Parse a player answer into a float (or None when it is not valid).

    Accepts decimal strings like "0.75" (including "0.750") and fraction
    strings like "3/4" by dividing the numerator by the denominator.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        if "/" in raw:
            parts = raw.split("/")
            if len(parts) != 2:
                return None
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
            if denominator == 0:
                return None
            return numerator / denominator
        return float(raw)
    except ValueError:
        return None


def choose_challenge(last_index):
    """Return a random challenge index, never the same as the previous one."""
    index = random.randrange(len(CHALLENGES))
    if index == last_index:
        index = (index + 1) % len(CHALLENGES)
    return index


class SpaceFractionsApp:
    """Tkinter application for the Space Fractions game."""

    def __init__(self, root):
        self.root = root
        self.level = 0
        self.score = 0
        self.streak = 0
        self.fraction = ""
        self.expected = 0.0
        self._last_index = -1
        self._build_ui()
        self.new_challenge()

    # -- layout -------------------------------------------------------------
    def _build_ui(self):
        self.root.title("Space Fractions")
        self.root.configure(bg=BG)
        self.root.geometry("520x620+60+40")
        self.root.minsize(460, 540)

        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=22, pady=18)

        tk.Label(outer, text="OBSERVATORY CHALLENGE", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(outer, text="Space Fractions", bg=BG, fg=INK,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(2, 0))
        tk.Label(outer, text="Convert the fraction to a decimal.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 14))

        panel = tk.Frame(outer, bg=PANEL, highlightbackground=LINE,
                         highlightthickness=1)
        panel.pack(fill="both", expand=True)

        body = tk.Frame(panel, bg=PANEL)
        body.pack(fill="both", expand=True, padx=18, pady=16)

        chips = tk.Frame(body, bg=PANEL)
        chips.pack(fill="x")
        self.level_var = tk.StringVar(value="LEVEL 0")
        self.score_var = tk.StringVar(value="SCORE 0")
        self.streak_var = tk.StringVar(value="STREAK 0")
        for var in (self.level_var, self.score_var, self.streak_var):
            tk.Label(chips, textvariable=var, bg=PANEL, fg=MUTED,
                     font=("Consolas", 10), bd=1, relief="solid",
                     padx=9, pady=3).pack(side="left", padx=(0, 6))

        self.fraction_var = tk.StringVar(value="--")
        tk.Label(body, textvariable=self.fraction_var, bg=PANEL, fg=INK,
                 font=("Georgia", 50, "bold")).pack(pady=(20, 4))

        tk.Label(body, text="What decimal value does this fraction equal?",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(pady=(0, 10))

        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x")
        self.entry = tk.Entry(row, bg="#0C1428", fg=INK, insertbackground=INK,
                              font=("Consolas", 13), relief="flat",
                              highlightbackground=LINE, highlightcolor=ACCENT,
                              highlightthickness=1)
        self.entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 8))
        self.entry.bind("<Return>", lambda event: self.check_answer())
        self.check_button = tk.Button(row, text="Check Answer",
                                      command=self.check_answer, bg=ACCENT,
                                      fg="#1A1408", activebackground=ACCENT_DIM,
                                      activeforeground="#1A1408", relief="flat",
                                      font=("Segoe UI", 10, "bold"), padx=14,
                                      pady=7, cursor="hand2")
        self.check_button.pack(side="left")

        self.feedback_var = tk.StringVar(value="")
        self.feedback_label = tk.Label(body, textvariable=self.feedback_var,
                                       bg=PANEL, fg=MUTED, font=("Segoe UI", 10),
                                       anchor="w", justify="left", wraplength=380)
        self.feedback_label.pack(fill="x", pady=(12, 0))

        self.next_button = tk.Button(body, text="Next Challenge",
                                     command=self.new_challenge, bg=PANEL,
                                     fg=MUTED, activebackground=PANEL,
                                     activeforeground=INK, relief="solid", bd=1,
                                     font=("Segoe UI", 10), padx=12, pady=6,
                                     cursor="hand2")
        self.next_button.pack(anchor="w", pady=(14, 0))

        tk.Label(body, text="Decimals (0.75) and fractions (3/4) are both accepted.",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(14, 0))

    # -- game flow ----------------------------------------------------------
    def new_challenge(self):
        self._last_index = choose_challenge(self._last_index)
        fraction, numerator, denominator = CHALLENGES[self._last_index]
        self.fraction = fraction
        self.expected = round(numerator / denominator, 3)
        self.level += 1
        self.fraction_var.set(fraction)
        self.level_var.set("LEVEL " + str(self.level))
        self._set_feedback("")
        self.entry.delete(0, "end")
        self.entry.focus_set()
        self.check_button.config(state="normal")
        self.next_button.config(state="normal")

    def check_answer(self):
        submitted = parse_answer(self.entry.get())
        if submitted is None:
            self._set_feedback("Type a decimal (0.75) or a fraction (3/4).", BAD)
            self.entry.focus_set()
            return
        if abs(submitted - self.expected) <= TOLERANCE:
            self.score += 100
            self.streak += 1
            self.score_var.set("SCORE " + str(self.score))
            self.streak_var.set("STREAK " + str(self.streak))
            self._set_feedback("Correct. " + self.fraction + " = " + self._fmt(self.expected) + ".", OK)
        else:
            self.streak = 0
            self.streak_var.set("STREAK 0")
            self._set_feedback("Not quite. " + self.fraction + " = " + self._fmt(self.expected) + ".", BAD)
        self.entry.delete(0, "end")
        self.entry.focus_set()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _fmt(value):
        text = ("%.3f" % value).rstrip("0").rstrip(".")
        return text if text else "0"

    def _set_feedback(self, text, color=MUTED):
        self.feedback_var.set(text)
        self.feedback_label.config(fg=color)


def main():
    root = tk.Tk()
    SpaceFractionsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
''',
    "requirements.txt": (
        "# Space Fractions - desktop game requirements\n"
        "#\n"
        "# No third-party packages: the game runs on the Python\n"
        "# standard library only (tkinter ships with the python.org installers).\n"
        "#\n"
        "# Just run:  python main_game.py\n"
    ),
    "schema.sql": (
        "-- Space Fractions Database Schema\n"
        "CREATE TABLE IF NOT EXISTS players (\n"
        "    player_id SERIAL PRIMARY KEY,\n"
        "    username VARCHAR(50) NOT NULL,\n"
        "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
        ");\n\n"
        "CREATE TABLE IF NOT EXISTS game_sessions (\n"
        "    session_id SERIAL PRIMARY KEY,\n"
        "    player_id INT REFERENCES players(player_id),\n"
        "    score INT DEFAULT 0,\n"
        "    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
        ");\n"
    ),
    "openapi.yaml": (
        "openapi: 3.0.0\n"
        "info:\n"
        "  title: Space Fractions Game (desktop)\n"
        "  version: 1.0.0\n"
        "  description: Documented interface of the desktop game operations.\n"
        "paths:\n"
        "  /api/v1/game/fractions/level:\n"
        "    get:\n"
        "      summary: Picks a new random fraction challenge and its expected decimal\n"
        "      responses:\n"
        "        '200':\n"
        "          description: OK\n"
        "  /api/v1/game/fractions/validate:\n"
        "    post:\n"
        "      summary: Validates an answer (decimals like 0.75 or fractions like 3/4)\n"
        "      responses:\n"
        "        '200':\n"
        "          description: OK\n"
    ),
    "Dockerfile": (
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "COPY . .\n"
        "# The game is a Tkinter GUI: run it on a desktop session with a display.\n"
        'CMD ["python", "main_game.py"]\n'
    ),
    "README.md": (
        "# Space Fractions - Desktop Game\n\n"
        "A standalone desktop game built with Python and Tkinter: read the\n"
        "fraction on screen, convert it to a decimal, and build your score\n"
        "and streak. Dark space theme, no web server required.\n\n"
        "## Run\n\n"
        "```bash\n"
        "python main_game.py\n"
        "```\n\n"
        "No third-party packages are needed (tkinter ships with Python).\n\n"
        "## How to play\n\n"
        "- A new random fraction appears on every challenge (1/2, 3/4, ...).\n"
        "- Type the decimal (0.75) or the fraction itself (3/4), then press\n"
        "  Check Answer.\n"
        "- Correct answers add 100 points and grow the streak.\n"
        "- A wrong answer shows the expected decimal and resets the streak.\n"
        "- Next Challenge rolls a fresh fraction and advances the level.\n\n"
        "## Project files\n\n"
        "- `main_game.py` - the complete game (Tkinter UI + game logic)\n"
        "- `requirements.txt` - dependency notes (standard library only)\n"
        "- `schema.sql` - example database schema (players, sessions)\n"
        "- `openapi.yaml` - documented interface of the game operations\n"
        "- `Dockerfile` - container packaging (needs a display for the GUI)\n"
    ),
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def sanitize_for_json(obj: Any) -> Any:
    """Make a structure safely JSON-serialisable (paths, sets, unicode dashes)."""
    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, str):
        return obj.replace("\u2014", "-").replace("\u2013", "-")
    return obj


def _de_fence(content: str) -> str:
    """Remove markdown code fences that wrap a generated file's content."""
    text = content.strip("\n")
    lines = text.split("\n") if text else []
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if re.fullmatch(r"```[A-Za-z0-9_.+-]*", lines[0].strip()):
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def _parse_json_mapping(text: str) -> List[Dict[str, str]]:
    """Fallback parser for responses shaped like a JSON file mapping."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []

    items: List[Dict[str, str]] = []
    if isinstance(data, dict) and isinstance(data.get("files"), list):
        for entry in data["files"]:
            if isinstance(entry, dict) and "path" in entry and "content" in entry:
                items.append(
                    {"path": str(entry["path"]), "content": str(entry["content"])}
                )
        return items
    if isinstance(data, dict) and isinstance(data.get("files"), dict):
        return [
            {"path": str(key), "content": str(value)}
            for key, value in data["files"].items()
            if isinstance(value, str)
        ]
    if isinstance(data, dict):
        candidate = {str(k): v for k, v in data.items() if isinstance(v, str)}
        path_like = [k for k in candidate if re.search(r"[/\\]|\.\w{1,6}$", k)]
        if candidate and len(path_like) == len(candidate):
            return [{"path": k, "content": candidate[k]} for k in candidate]
    return []


def _salvage_blocks(text: str) -> List[Dict[str, str]]:
    """Recover files from a truncated response by closing open tag streams."""
    files: List[Dict[str, str]] = []
    parts = _FILE_START_RE.split(text)
    for part in parts[1:]:
        header, _, rest = part.partition("\n")
        rel_path = header.strip().rstrip("=").strip().strip("`").strip()
        if not rel_path:
            continue
        content = rest
        end_index = content.find("===END FILE===")
        if end_index != -1:
            content = content[:end_index]
        content = content.strip("\n")
        if not content:
            continue
        files.append(
            {"path": rel_path, "content": content if content.endswith("\n") else content + "\n"}
        )
    return files


def parse_generated_files(model_output: str) -> List[Dict[str, str]]:
    """Extract ``{"path", "content"}`` entries from a model response.

    Primary format is the strict ``===FILE: <path>=== ... ===END FILE===``
    block format.  Truncated responses (uneven markers) are salvaged instead
    of raising; a JSON file-mapping fallback is also accepted.
    """
    text = (model_output or "").replace("\r\n", "\n").replace("\r", "\n")

    starts = len(_FILE_START_RE.findall(text))
    ends = len(_FILE_END_RE.findall(text))

    if starts and starts == ends:
        files: List[Dict[str, str]] = []
        for match in _FILE_BLOCK_RE.finditer(text):
            rel_path = match.group("path").strip().strip("`").strip()
            if rel_path:
                files.append(
                    {"path": rel_path, "content": _de_fence(match.group("content"))}
                )
        if files:
            return files
        raise ValueError("file markers found but no complete blocks could be parsed.")

    if starts and starts != ends:
        # token truncation: salvage complete blocks + close the open stream
        salvaged = _salvage_blocks(text)
        if salvaged:
            return salvaged
        raise ValueError(
            f"uneven markers ({starts} vs {ends}) and no content could be salvaged."
        )

    files = _parse_json_mapping(text)
    if files:
        return files

    raise ValueError(
        "no '===FILE: <path>===' blocks (and no JSON file mapping) found in the "
        f"model response. Excerpt: {text[:500]!r}"
    )


def _restrict_to_core_files(
    files: Sequence[Dict[str, str]], log: Callable[[str], None]
) -> List[Dict[str, str]]:
    """Keep only the essential core files, flattened to their canonical paths.

    Nested paths are remapped to their core file names (src/main.py ->
    main_game.py, docs/readme.md -> README.md, ...).  Anything outside the
    core set is skipped and logged, so no other sub-directories are
    created.
    """
    kept: Dict[str, str] = {}
    order: List[str] = []
    for entry in files:
        raw = str(entry.get("path", "")).replace("\\", "/").strip()
        if not raw:
            continue
        flat = raw.rsplit("/", 1)[-1].lower()
        canonical = _CORE_LOOKUP.get(flat)
        if canonical is None:
            log(f"[agent] skipped extra file: {raw} (outside the core file set)")
            continue
        if canonical not in kept:
            order.append(canonical)
        kept[canonical] = str(entry.get("content", ""))
    return [{"path": name, "content": kept[name]} for name in order]


def _safe_output_path(root: Path, relative_path: str) -> Path:
    """Join ``relative_path`` under ``root``, refusing any escape attempt."""
    cleaned = relative_path.replace("\\", "/").strip()
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid generated file path: {relative_path!r}")
    if re.match(r"^[A-Za-z]:", cleaned) or cleaned.startswith("/"):
        raise ValueError(f"Absolute paths are not allowed: {relative_path!r}")

    root_resolved = root.resolve()
    candidate = (root_resolved / cleaned).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"Path escapes the output directory: {relative_path!r}")
    return candidate


def write_generated_files(
    files: Sequence[Dict[str, str]], output_dir: Path
) -> List[str]:
    """Write parsed files below ``output_dir``; returns the relative paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()

    written: List[str] = []
    for entry in files:
        target = _safe_output_path(output_dir_resolved, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        content = entry["content"]
        if content and not content.endswith("\n"):
            content += "\n"
        target.write_text(content, encoding="utf-8", newline="\n")
        written.append(str(target.relative_to(output_dir_resolved)).replace("\\", "/"))
    return written


def _write_static_project_files(
    output_dir: Path,
    payload: Optional[Dict[str, Any]],
    log: Callable[[str], None],
    skip_existing: bool = True,
    only: Optional[Sequence[str]] = None,
) -> List[str]:
    """Write the static backend + game UI file set (recovery / completion)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contents: Dict[str, str] = dict(_STATIC_FILES)
    contents["architecture_payload.json"] = json.dumps(
        sanitize_for_json(payload if payload is not None else {}),
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    wanted = set(only) if only is not None else None
    created: List[str] = []
    for name in _STATIC_FALLBACK_ORDER:
        if wanted is not None and name not in wanted:
            continue
        target = output_dir / name
        if skip_existing and target.exists():
            log(f"[agent] kept existing: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = contents[name]
            if not content.endswith("\n"):
                content += "\n"
            target.write_text(content, encoding="utf-8", newline="\n")
            log(f"[agent] Created: {name}")
        created.append(str(target.resolve()))
    return created


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #

def build_system_prompt(project_name: str = "Space Fractions") -> str:
    """Return the system prompt that drives the file generation."""
    return f"""You are CodeAgent, a senior Python developer and DevOps engineer.
You turn architecture documentation into a complete, production-ready,
error-free standalone desktop application for "{project_name}": a dark
space-themed Tkinter game (no web server, no browser UI).

## Deliverables
Generate all of the following across the requested stages:
{_REQUIRED_FILES}
## Validation requirements (game logic)
- The Check Answer action must accept the player's answer robustly: trim
  whitespace; accept decimal strings like "0.75" (including equivalents
  such as "0.750"); and accept fraction answers like "3/4" by dividing
  numerator by denominator and comparing numerically with a small float
  tolerance (about 0.000001).
- Every new challenge (on startup and on Next Challenge) must pick a
  brand-new random fraction from a pool of at least eight (e.g. 1/2, 3/4,
  2/5, 5/8, 1/4, 4/5, 3/10, 9/10), compute the expected decimal dynamically
  from numerator / denominator at that moment, advance the level counter,
  and never repeat the same fraction twice in a row.
- Score, streak and level are tracked in the window: +100 points and a
  growing streak for a correct answer; a wrong answer shows the expected
  decimal and resets the streak.

## Output format (STRICT)
- Emit every file as a block in exactly this shape:
===FILE: <relative/path/to/file>===
<complete file content>
===END FILE===
- Only the core files are expected (all at the project root). Never emit
  server.js, package.json, public/index.html, Express, Flask or any other
  web files - this is a desktop application. Files outside the core set
  are ignored.
- Never wrap file contents in markdown code fences and never truncate a
  file; no "..." or "TODO" placeholders - full content only.

## Quality bar (must all hold)
- Production-ready and error-free. No syntax errors: every opening brace
  has a matching closing brace, every string literal is correctly quoted,
  and every import exists (standard library only for main_game.py - no pip
  installs may be required to run it). JSON, YAML and SQL must be valid.
- Fully consistent: the game logic, the README run instructions and the
  Dockerfile must all describe the same single-file Tkinter desktop game
  (python main_game.py); there is no web server and there are no HTTP
  calls anywhere in the project.
- The game window must be complete and usable: dark space-themed; score,
  streak and level counters; a large fraction display that is filled
  immediately from the game state (a placeholder like -- must never get
  stuck); a bound Check Answer button; a Next Challenge action; an answer
  field that is cleared and buttons that stay enabled after every action;
  and a status message that reports correct/incorrect answers including
  the expected decimal when wrong.
"""


def build_stage_messages(
    payload: Dict[str, Any],
    *,
    stage_name: str,
    instruction: str,
    already_generated: Sequence[str],
) -> List[Dict[str, str]]:
    """Build the [system, user] message list for one generation stage."""
    project_name = str(payload.get("project") or "Space Fractions")
    system_prompt = build_system_prompt(project_name)

    parts: List[str] = []
    parts.append("## Architecture payload (JSON)")
    parts.append(
        json.dumps(sanitize_for_json(payload), ensure_ascii=False, separators=(",", ":"))
    )
    parts.append("")
    parts.append(f"## Stage: {stage_name}")
    parts.append(instruction)
    if already_generated:
        parts.append("")
        parts.append(
            "## Files already generated (keep them consistent; only re-emit "
            "a file if you need to change it)"
        )
        parts.extend(f"- {path}" for path in already_generated)
    parts.append("")
    parts.append(
        "Respond with the file blocks only, using the strict "
        "===FILE: <path>=== ... ===END FILE=== format."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


# --------------------------------------------------------------------------- #
# DeepSeek API client
# --------------------------------------------------------------------------- #

class DeepSeekClient:
    """Minimal client for the official DeepSeek chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEEPSEEK_API_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        key = (api_key or "").strip()
        key = "".join(ch for ch in key if ch.isprintable() and ch != " ")
        if not key:
            raise ValueError(
                "A DeepSeek API key is required for generation. Enter it in "
                f"the GUI or set the {ENV_API_KEY} environment variable."
            )
        self.api_key = key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Call /chat/completions and return {"content", "usage", "model"}."""
        request_body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.base_url, headers=headers, json=request_body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._sleep_before_retry(attempt, None)
                continue

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices") or []
                content = choices[0].get("message", {}).get("content") if choices else None
                if not content:
                    raise RuntimeError(
                        f"DeepSeek returned an empty response: {str(data)[:500]}"
                    )
                return {
                    "content": content,
                    "usage": data.get("usage", {}),
                    "model": data.get("model", self.model),
                }

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                self._sleep_before_retry(attempt, response.headers.get("Retry-After"))
                continue

            raise RuntimeError(
                f"DeepSeek API error (HTTP {response.status_code}): {response.text[:800]}"
            )

        raise RuntimeError(
            f"DeepSeek API request failed after {self.max_retries} attempts: {last_error}"
        )

    def _sleep_before_retry(self, attempt: int, retry_after: Optional[str]) -> None:
        if attempt >= self.max_retries:
            return
        delay = min(2 ** attempt, 30.0)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _normalize_call(
    args: Sequence[Any], kwargs: Dict[str, Any]
) -> "tuple[Optional[Callable[[str], None]], Optional[str], Optional[Dict[str, Any]], Optional[Any]]":
    """Accept both modern and legacy call signatures.

    Modern:  generate_project(payload, output_dir, api_key=api_key, log=...)
    Legacy:  generate_project_code(api_key=KEY, payload, output_dir, log_callback=...)
    """
    log_callback = kwargs.get("log") or kwargs.get("log_callback")
    api_key = kwargs.get("api_key")
    payload = (
        kwargs.get("parsed_architecture")
        or kwargs.get("architecture")
        or kwargs.get("payload")
    )
    output_dir = (
        kwargs.get("output_dir")
        or kwargs.get("output_folder")
        or kwargs.get("target_dir")
        or kwargs.get("out_dir")
    )

    for value in args:
        if value is None:
            continue
        if payload is None and isinstance(value, dict):
            payload = value
        elif output_dir is None and isinstance(value, Path):
            output_dir = value
        elif output_dir is None and isinstance(value, str) and (
            re.search(r"[/\\]", value)
            or os.path.isabs(value)
            or value.lower().endswith((".md", ".json"))
        ):
            output_dir = value
        elif api_key is None and isinstance(value, str):
            api_key = value

    return log_callback, api_key, payload, output_dir


def _request_stage_files(
    client: DeepSeekClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    log: Callable[[str], None],
    stage_name: str,
    attempts: int = 2,
) -> List[Dict[str, str]]:
    """Ask the API for one stage of files, retrying once on format problems."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        result = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return parse_generated_files(result["content"])
        except ValueError as exc:
            last_exc = exc
            log(
                f"[agent] stage '{stage_name}': response format issue "
                f"(attempt {attempt}/{attempts}): {exc}"
            )
    raise last_exc if last_exc else RuntimeError("stage failed")


def generate_project_code(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Generate the desktop game project files with the live DeepSeek API.

    The output covers the Tkinter desktop game (main_game.py), the SQL
    schema, OpenAPI spec, Dockerfile, README and the payload copy, so the
    reported file count is stable.  Truncated responses are recovered
    automatically (salvage + high-quality static completion) and the
    summary always contains a non-zero ``files_written`` list.
    """
    log_callback, api_key, payload, output_dir = _normalize_call(args, kwargs)

    def log(message: str) -> None:
        if log_callback:
            try:
                log_callback(message)
            except Exception:
                pass
        print(message)

    if not payload:
        raise ValueError(
            "generate_project: the parsed architecture payload is required "
            "(pass it as the first argument or as payload=...)."
        )

    if output_dir is None:
        output_dir = Path.cwd() / "output"
    output_dir = Path(str(output_dir))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create output directory '{output_dir}': {exc}") from exc

    model = str(kwargs.get("model") or DEFAULT_MODEL)
    temperature = float(kwargs.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(kwargs.get("max_tokens", DEFAULT_MAX_TOKENS))
    timeout = int(kwargs.get("timeout", DEFAULT_TIMEOUT))
    stages: Sequence[Dict[str, str]] = kwargs.get("stages") or DEFAULT_STAGES

    log("[agent] Initializing DeepSeek code-generation agent ...")
    log(f"[agent] Model: {model}")
    log(f"[agent] Output directory: {output_dir.resolve()}")

    # ---- API client (or injected test client) ------------------------------
    client = kwargs.get("client")
    if client is None:
        key = (api_key or os.environ.get(ENV_API_KEY, "") or "").strip()
        if not key:
            raise ValueError(
                "A DeepSeek API key is required for generation. Enter it in "
                "the GUI and try again."
            )
        client = DeepSeekClient(api_key=key, model=model, timeout=timeout)

    # ---- staged generation --------------------------------------------------
    written_all: List[str] = []
    stage_reports: List[Dict[str, Any]] = []
    used_fallback = False

    for index, spec in enumerate(stages, start=1):
        stage_name = str(spec.get("name", f"stage-{index}"))
        instruction = str(spec.get("instruction") or spec.get("description") or "")
        log(f"[agent_core] stage '{stage_name}': requesting files from {model} ...")
        messages = build_stage_messages(
            payload,
            stage_name=stage_name,
            instruction=instruction,
            already_generated=written_all,
        )
        try:
            files = _request_stage_files(
                client, messages, temperature, max_tokens, log, stage_name
            )
            files = _restrict_to_core_files(files, log)
            if not files:
                raise ValueError("the response contained no core project files.")
            stage_files = write_generated_files(files, output_dir)
        except ValueError as exc:
            # truncated / unusable response - recover instead of crashing
            log(
                f"[agent] WARNING: stage '{stage_name}' produced an incomplete "
                f"response: {exc}"
            )
            log("[agent] Recovering: completing the project with fallback files ...")
            recovered = _write_static_project_files(output_dir, payload, log)
            for path in recovered:
                if path not in written_all:
                    written_all.append(path)
            stage_reports.append({"stage": stage_name, "files": recovered, "recovered": True})
            used_fallback = True
            break
        except Exception as exc:  # noqa: BLE001 - surface real failures (HTTP etc.)
            message = f"stage '{stage_name}' failed: {type(exc).__name__}: {exc}"
            log(f"[agent] ERROR: {message}")
            raise RuntimeError(message) from exc

        for path in stage_files:
            if path not in written_all:
                written_all.append(path)
        stage_reports.append({"stage": stage_name, "files": stage_files})
        log(f"[agent_core] stage '{stage_name}': wrote {len(stage_files)} file(s)")

    # ---- ensure the complete core set is present ------------------------------
    missing = [name for name in CORE_PROJECT_FILES if not (output_dir / name).exists()]
    if missing:
        log(f"[agent] Completing missing core file(s): {', '.join(missing)}")
        completed = _write_static_project_files(
            output_dir, payload, log, skip_existing=True, only=missing
        )
        for path in completed:
            if path not in written_all:
                written_all.append(path)

    # ---- payload copy for traceability ---------------------------------------
    payload_path = output_dir / "architecture_payload.json"
    try:
        payload_path.write_text(
            json.dumps(sanitize_for_json(payload), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        log(f"[agent] Architecture payload saved to: {payload_path}")
        if str(payload_path.resolve()) not in written_all:
            written_all.append(str(payload_path.resolve()))
    except OSError as exc:
        log(f"[agent] WARNING: could not save architecture payload: {exc}")

    log(f"[agent] SUCCESS: Generated {len(written_all)} file(s) into {output_dir}")
    return {
        "output_dir": str(output_dir),
        "files_written": written_all,
        "stages": stage_reports,
        "used_fallback": used_fallback,
        "mode": "api",
        "model": model,
    }


# Backwards-compatible alias
generate_project = generate_project_code


if __name__ == "__main__":
    print("agent_core.py is a library module - launch the GUI with: python main_gui.py")
