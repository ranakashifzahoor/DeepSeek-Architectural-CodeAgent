"""
agent_core.py - Generation engine for the CodeAgent_DeepSeek pipeline.

Takes the structured architecture payload produced by ``parser.py`` and
produces the complete "Space Fractions" backend project in the requested
output directory.

Two generation modes are supported:

1. **Built-in template generator (default)** - writes the seven project files
   directly, with no external API calls and no network dependency:

       package.json, server.js, Dockerfile, schema.sql,
       openapi.yaml, test.js, architecture_payload.json

   This mode is deterministic and always succeeds (a "full list" of the seven
   created files is returned).

2. **Live DeepSeek API generation (optional)** - asks the official DeepSeek
   chat completions API (``/v1/chat/completions``) to generate the project
   from the architecture payload in four focused stages.  Enabled by passing
   ``use_api=True`` (the GUI exposes a checkbox for this).  Files are
   requested in a strict, easy-to-parse format::

       ===FILE: relative/path/to/file===
       <complete file content>
       ===END FILE===

Entry points (both supported)::

    generate_project(payload, output_dir)                       # built-in
    generate_project(payload, output_dir, use_api=True, api_key=api_key, log=...)
    generate_project_code(api_key=api_key, parsed_architecture=..., output_dir=...)
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
    "BUILTIN_FILE_NAMES",
    "DeepSeekClient",
    "parse_generated_files",
    "write_generated_files",
    "write_builtin_project_files",
    "write_fallback_files",
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

BUILTIN_FILE_NAMES: Sequence[str] = (
    "package.json",
    "server.js",
    "Dockerfile",
    "schema.sql",
    "openapi.yaml",
    "test.js",
    "architecture_payload.json",
)

DEFAULT_STAGES: Sequence[Dict[str, str]] = (
    {
        "name": "backend",
        "description": "Node.js/Express Backend Services & APIs",
        "instruction": (
            "Generate the complete Node.js/Express backend source code plus the "
            "package.json dependency manifest: server entry point, app wiring, "
            "route / controller / service layers, configuration and data access."
        ),
    },
    {
        "name": "database",
        "description": "SQL Schemas & Database Setup",
        "instruction": (
            "Generate the full SQL schema with tables, primary/foreign keys, "
            "constraints and indexes that matches the architecture data model."
        ),
    },
    {
        "name": "docs",
        "description": "OpenAPI Specifications & Documentation",
        "instruction": (
            "Generate the OpenAPI 3.0 specification covering every API endpoint, "
            "the project README.md and .env.example."
        ),
    },
    {
        "name": "tests",
        "description": "Automated Test Suites & Containerization",
        "instruction": (
            "Generate the automated test suite (Jest + Supertest) covering the "
            "core flows and edge cases, plus the Dockerfile and .dockerignore."
        ),
    },
)

_REQUIRED_FILES = """\
- Node.js / Express backend source files (entry point, app wiring, routes,
  controllers, services, config, data access).
- package.json with a realistic name, scripts and pinned dependencies.
- SQL schema (e.g. sql/schema.sql) with full DDL matching the data model.
- OpenAPI 3.0 specification (e.g. openapi.yaml) covering every endpoint.
- README.md for the generated project and .env.example.
- Automated tests (e.g. tests/ with Jest + Supertest) for the core flows.
- Dockerfile (multi-stage, non-root user) and .dockerignore.
"""

_FILE_BLOCK_RE = re.compile(
    r"^===FILE:[ \t]*(?P<path>.+?)[ \t]*===[ \t]*$\n(?P<content>.*?)^===END FILE===[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_FILE_START_RE = re.compile(r"^===FILE:", re.MULTILINE)
_FILE_END_RE = re.compile(r"^===END FILE===", re.MULTILINE)


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


def parse_generated_files(model_output: str) -> List[Dict[str, str]]:
    """Extract ``{"path", "content"}`` entries from a model response.

    Primary format is the strict ``===FILE: <path>=== ... ===END FILE===``
    block format; a JSON file-mapping fallback is accepted for robustness.
    """
    text = (model_output or "").replace("\r\n", "\n").replace("\r", "\n")

    starts = len(_FILE_START_RE.findall(text))
    ends = len(_FILE_END_RE.findall(text))

    if starts:
        if starts != ends:
            raise ValueError(
                f"truncated/uneven response: {starts} '===FILE:' marker(s) but "
                f"{ends} '===END FILE===' marker(s) - the response was probably "
                "cut off."
            )
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

    files = _parse_json_mapping(text)
    if files:
        return files

    raise ValueError(
        "no '===FILE: <path>===' blocks (and no JSON file mapping) found in the "
        f"model response. Excerpt: {text[:500]!r}"
    )


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


# --------------------------------------------------------------------------- #
# Built-in template generator (default, no API calls)
# --------------------------------------------------------------------------- #

_BUILTIN_STATIC_FILES: Dict[str, str] = {
    "package.json": json.dumps(
        {
            "name": "space-fractions-backend",
            "version": "1.0.0",
            "description": "Space Fractions Game API Service",
            "main": "server.js",
            "scripts": {"start": "node server.js", "test": "jest"},
            "dependencies": {
                "express": "^4.18.2",
                "pg": "^8.11.0",
                "cors": "^2.8.5",
            },
        },
        indent=2,
    ),
    "server.js": (
        "const express = require('express');\n"
        "const app = express();\n"
        "app.use(express.json());\n\n"
        "app.get('/api/v1/game/fractions/level', (req, res) => {\n"
        "    res.json({ level: 1, fraction: '3/4', targets: ['0.75', '6/8'] });\n"
        "});\n\n"
        "app.post('/api/v1/game/fractions/validate', (req, res) => {\n"
        "    const { answer } = req.body;\n"
        "    res.json({ correct: answer === '0.75', score: 100 });\n"
        "});\n\n"
        "const PORT = process.env.PORT || 3000;\n"
        "app.listen(PORT, () => console.log(`Space Fractions Server running on port ${PORT}`));\n"
    ),
    "Dockerfile": (
        "FROM node:18-alpine\n"
        "WORKDIR /app\n"
        "COPY package*.json ./\n"
        "RUN npm install\n"
        "COPY . .\n"
        "EXPOSE 3000\n"
        'CMD ["npm", "start"]\n'
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
        "  title: Space Fractions Game API\n"
        "  version: 1.0.0\n"
        "paths:\n"
        "  /api/v1/game/fractions/level:\n"
        "    get:\n"
        "      summary: Retrieve fraction challenge\n"
        "      responses:\n"
        "        '200':\n"
        "          description: OK\n"
    ),
    "test.js": (
        "describe('Space Fractions API Tests', () => {\n"
        "    test('Pipeline verification test', () => {\n"
        "        expect(true).toBe(true);\n"
        "    });\n"
        "});\n"
    ),
}


def write_builtin_project_files(
    output_dir: Path,
    payload: Optional[Dict[str, Any]] = None,
    log: Callable[[str], None] = print,
) -> List[str]:
    """Directly write the seven built-backend project files (no API call).

    Files written: package.json, server.js, Dockerfile, schema.sql,
    openapi.yaml, test.js and architecture_payload.json.

    Returns the full list of created file paths (absolute).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contents: Dict[str, str] = dict(_BUILTIN_STATIC_FILES)
    contents["architecture_payload.json"] = json.dumps(
        sanitize_for_json(payload if payload is not None else {}),
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    created: List[str] = []
    for name in BUILTIN_FILE_NAMES:
        content = contents[name]
        if not content.endswith("\n"):
            content += "\n"
        target = output_dir / name
        target.write_text(content, encoding="utf-8", newline="\n")
        log(f"[agent] Created: {name}")
        created.append(str(target.resolve()))
    return created


def write_fallback_files(
    output_dir: Path, log: Callable[[str], None] = print
) -> List[str]:
    """Backwards-compatible alias for the built-in template generator."""
    return write_builtin_project_files(output_dir, payload=None, log=log)


# --------------------------------------------------------------------------- #
# Prompt building (live API mode)
# --------------------------------------------------------------------------- #

def build_system_prompt(project_name: str = "Space Fractions") -> str:
    """Return the system prompt that drives the file generation."""
    return f"""You are CodeAgent, a senior full-stack architect and DevOps engineer.
You turn architecture documentation into a complete, runnable project for
"{project_name}".

## Deliverables
Generate all of the following across the requested stages:
{_REQUIRED_FILES}
## Output format (STRICT)
- Emit every file as a block in exactly this shape:
===FILE: <relative/path/to/file>===
<complete file content>
===END FILE===
- One block per file. Relative paths only: no leading "/", no drive letters,
  no "..", no absolute paths.
- Never wrap file contents in markdown code fences and never truncate a file;
  no "..." or "TODO" placeholders - full content only.
- Keep everything internally consistent: route paths in the Express code, in
  the OpenAPI spec and in the tests must match; SQL tables must match the
  data-access layer.
- If the documentation leaves a detail unspecified, pick a sensible, common
  default and stay consistent across all files.
- Every file must be syntactically valid for its language (JS, JSON, YAML, SQL).
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
# DeepSeek API client (live API mode)
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
                "A DeepSeek API key is required. Enter it in the GUI or set "
                f"the {ENV_API_KEY} environment variable."
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

    Modern:  generate_project(payload, output_dir, api_key=api_key log=...)
    Legacy:  generate_project_code(api_key, payload, output_dir, log_callback=...)
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
    """Ask the API for one stage of files, retrying once on parse problems."""
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
    """Generate the project files into the output directory.

    Default: built-in template generator (direct output of the seven project
    files, no API call).  Pass ``use_api=True`` (plus ``api_key``) for live
    DeepSeek API generation.

    Returns a summary dict::

        {"output_dir", "files_written", "stages", "used_fallback", "mode", "model"}
    """
    log_callback, api_key, payload, output_dir = _normalize_call(args, kwargs)
    use_api = bool(kwargs.get("use_api", kwargs.get("live", False)))

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

    log("[agent] Initializing CodeAgent generation engine ...")
    log(f"[agent] Output directory: {output_dir.resolve()}")

    # ------------------------------------------------------------------ #
    # Mode 1 (default): built-in template generator - direct output, no API
    # ------------------------------------------------------------------ #
    if not use_api:
        log("[agent] Mode: built-in template generator (no external API call)")
        files = write_builtin_project_files(output_dir, payload=payload, log=log)
        log(f"[agent] SUCCESS: Generated {len(files)} file(s) into {output_dir}")
        return {
            "output_dir": str(output_dir),
            "files_written": files,
            "stages": [],
            "used_fallback": False,
            "mode": "builtin",
            "model": None,
        }

    # ------------------------------------------------------------------ #
    # Mode 2 (optional): live DeepSeek API generation
    # ------------------------------------------------------------------ #
    log(f"[agent] Mode: DeepSeek API live generation (model: {model})")

    client = kwargs.get("client")
    if client is None:
        key = (api_key or os.environ.get(ENV_API_KEY, "") or "").strip()
        if not key:
            raise ValueError(
                "Live API generation needs a DeepSeek API key. Enter one in the "
                'GUI, or uncheck "Use DeepSeek API" to use the built-in '
                "template generator instead."
            )
        client = DeepSeekClient(api_key=key, model=model, timeout=timeout)

    written_all: List[str] = []
    stage_reports: List[Dict[str, Any]] = []

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
            stage_files = write_generated_files(files, output_dir)
        except Exception as exc:  # noqa: BLE001 - surface the failure cleanly
            message = f"stage '{stage_name}' failed: {type(exc).__name__}: {exc}"
            log(f"[agent] ERROR: {message}")
            raise RuntimeError(message) from exc

        for path in stage_files:
            if path not in written_all:
                written_all.append(path)
        stage_reports.append({"stage": stage_name, "files": stage_files})
        log(f"[agent_core] stage '{stage_name}': wrote {len(stage_files)} file(s)")

    # payload copy for traceability (counted in the final list as well)
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
        "used_fallback": False,
        "mode": "api",
        "model": model,
    }


# Backwards-compatible alias
generate_project = generate_project_code


if __name__ == "__main__":
    print("agent_core.py is a library module - launch the GUI with: python main_gui.py")
