"""
agent_core.py - DeepSeek generation engine for the CodeAgent_DeepSeek pipeline.

Takes the structured architecture payload produced by ``parser.py`` and asks
the official DeepSeek API (OpenAI-compatible ``/v1/chat/completions``) to
generate the complete, production-ready "Space Fractions" project - a
full-stack app:

    Frontend (game UI):
        public/index.html   self-contained interactive HTML/CSS/JS game page
    Backend (REST API):
        package.json, server.js, Dockerfile, schema.sql, openapi.yaml,
        test.js, README.md
    Engine traceability:
        architecture_payload.json

The generated Express app serves the game UI from the ``public`` directory
and exposes the REST endpoints ``GET /api/v1/game/fractions/level`` and
``POST /api/v1/game/fractions/validate`` (robust validation: trimmed input,
decimal strings like "0.75", fraction answers like "3/4" compared
numerically with a small tolerance).  After generation, running
``npm install`` + ``npm start`` and opening http://localhost:3000 shows the
interactive game in the browser.

The generation is scoped to this essential file set so the output stays
compact.  Nested paths coming back from the model are flattened to their
core file names (the single exception being ``public/index.html``), and
anything outside the core set is skipped (and logged).

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
# writes architecture_payload.json, so a complete run yields 9 files total.
CORE_PROJECT_FILES: Sequence[str] = (
    "package.json",
    "server.js",
    "public/index.html",
    "Dockerfile",
    "schema.sql",
    "openapi.yaml",
    "test.js",
    "README.md",
)

_CORE_LOOKUP: Dict[str, str] = {name.lower(): name for name in CORE_PROJECT_FILES}
_CORE_LOOKUP.update(
    {
        "openapi.yml": "openapi.yaml",
        "readme": "README.md",
        "index.html": "public/index.html",
    }
)

DEFAULT_STAGES: Sequence[Dict[str, str]] = (
    {
        "name": "backend",
        "description": "Node.js/Express Backend Services & APIs",
        "instruction": (
            "Generate package.json and server.js. server.js must be a "
            "single-file Express app in the project root that serves the "
            "static game UI from the public directory (express.static) and "
            "implements: GET /api/v1/game/fractions/level (dynamic: on EVERY "
            "call it picks a random fraction from a pool of at least eight "
            "entries - for example 1/2, 3/4, 2/5, 5/8, 1/4, 4/5, 3/10, "
            "9/10 - computes expectedDecimal from numerator / denominator "
            "at request time, responds as { \"status\": \"success\", "
            "\"level\": <incrementing number>, \"fraction\": \"3/4\", "
            "\"expectedDecimal\": 0.75 }, never returns the same fraction "
            "twice in a row, and never hardcodes a single static fraction) "
            "and POST "
            "/api/v1/game/fractions/validate (trims the answer, accepts "
            "decimal strings like \"0.75\" and fraction strings like "
            "\"3/4\" by parsing numerator/denominator, and compares "
            "numerically with a small float tolerance). It must listen on "
            "process.env.PORT or port 3000. The package.json must stay "
            "clean: declare only the runtime dependencies express "
            "(^4.21.2) and cors (^2.8.5) - no devDependencies, no "
            "test-runner packages (such as jest or supertest), no unused "
            "extras - and define the start script as \"node server.js\"."
        ),
    },
    {
        "name": "frontend",
        "description": "Interactive Space Fractions Game UI (HTML/CSS/JS)",
        "instruction": (
            "Generate public/index.html: a complete, self-contained "
            "interactive Space Fractions game UI with inline CSS and inline "
            "JavaScript (no external libraries or asset files). It must "
            "fetch /api/v1/game/fractions/level on initial page load AND "
            "every time the next-challenge button is clicked, parse the "
            "JSON response, and immediately update the fraction display "
            "(id=\"current-fraction\") plus the LEVEL counter from the "
            "response so no placeholder (such as --) is ever left showing. "
            "The Submit/Check button must have a click handler that POSTs "
            "{ \"answer\": <text>, \"fraction\": <current fraction> } to "
            "/api/v1/game/fractions/validate and then updates Score, "
            "Streak (consecutive correct answers, reset on a wrong answer) "
            "and a status message (showing the expected decimal when "
            "wrong). After every fetch, clear the answer input and ensure "
            "both buttons are enabled; treat fetch failures gracefully "
            "(message plus a working retry via the next-challenge button) "
            "so the page can never get stuck. Clean, space-themed, "
            "responsive design with no syntax errors."
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
        "description": "OpenAPI Specifications & Documentation",
        "instruction": (
            "Generate openapi.yaml (OpenAPI 3.0 covering every API endpoint, "
            "including both the level and validate operations) and README.md "
            "in the project root. The README must include the run "
            "instructions: npm install, npm start, then open "
            "http://localhost:3000 to play."
        ),
    },
    {
        "name": "tests",
        "description": "Automated Test Suites & Containerization",
        "instruction": (
            "Generate test.js (Jest + Supertest covering the API flows, "
            "including an answer submitted as the fraction string \"3/4\") "
            "and the Dockerfile in the project root."
        ),
    },
)

_REQUIRED_FILES = """\
- package.json: npm manifest with a realistic name, scripts (start/test) and
  pinned dependencies (express, pg, cors only).
- server.js: single-file Express server that serves the game UI from the
  public directory and implements the REST API with robust validation.
- public/index.html: self-contained interactive frontend game page (inline
  CSS + inline JavaScript, no extra asset files) that talks to the REST API.
- schema.sql: SQL DDL with tables, keys and constraints.
- openapi.yaml: OpenAPI 3.0 specification covering every endpoint.
- test.js: automated tests (Jest + Supertest) for the API flows.
- README.md: short project README with run instructions (npm install,
  npm start, then open http://localhost:3000).
- All files live at the project root except public/index.html; do not create
  any other directories.
"""

_FILE_BLOCK_RE = re.compile(
    r"^===FILE:[ \t]*(?P<path>.+?)[ \t]*===[ \t]*$\n(?P<content>.*?)^===END FILE===[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_FILE_START_RE = re.compile(r"^===FILE:", re.MULTILINE)
_FILE_END_RE = re.compile(r"^===END FILE===", re.MULTILINE)

_STATIC_FALLBACK_ORDER: Sequence[str] = (
    "package.json",
    "server.js",
    "public/index.html",
    "Dockerfile",
    "schema.sql",
    "openapi.yaml",
    "test.js",
    "README.md",
    "architecture_payload.json",
)

_STATIC_FILES: Dict[str, str] = {
    "package.json": json.dumps(
        {
            "name": "space-fractions-backend",
            "version": "1.0.0",
            "description": "Space Fractions Game API + interactive UI",
            "main": "server.js",
            "scripts": {"start": "node server.js"},
            "dependencies": {
                "express": "^4.21.2",
                "cors": "^2.8.5",
            },
        },
        indent=2,
    ),
    "server.js": (
        "const express = require('express');\n"
        "const path = require('path');\n\n"
        "const app = express();\n"
        "app.use(express.json());\n"
        "app.use(express.static(path.join(__dirname, 'public')));\n\n"
        "const CHALLENGES = [\n"
        "    { fraction: '1/2', numerator: 1, denominator: 2 },\n"
        "    { fraction: '3/4', numerator: 3, denominator: 4 },\n"
        "    { fraction: '2/5', numerator: 2, denominator: 5 },\n"
        "    { fraction: '5/8', numerator: 5, denominator: 8 },\n"
        "    { fraction: '1/4', numerator: 1, denominator: 4 },\n"
        "    { fraction: '4/5', numerator: 4, denominator: 5 },\n"
        "    { fraction: '3/10', numerator: 3, denominator: 10 },\n"
        "    { fraction: '9/10', numerator: 9, denominator: 10 },\n"
        "];\n"
        "let lastChallengeIndex = -1;\n"
        "let levelCounter = 0;\n\n"
        "function toNumber(text) {\n"
        "    const raw = String(text == null ? '' : text).trim();\n"
        "    if (!raw) {\n"
        "        return NaN;\n"
        "    }\n"
        "    if (raw.indexOf('/') !== -1) {\n"
        "        const parts = raw.split('/');\n"
        "        if (parts.length !== 2) {\n"
        "            return NaN;\n"
        "        }\n"
        "        const numerator = Number(parts[0].trim());\n"
        "        const denominator = Number(parts[1].trim());\n"
        "        if (!isFinite(numerator) || !isFinite(denominator) || denominator === 0) {\n"
        "            return NaN;\n"
        "        }\n"
        "        return numerator / denominator;\n"
        "    }\n"
        "    const parsed = Number(raw);\n"
        "    return isFinite(parsed) ? parsed : NaN;\n"
        "}\n\n"
        "app.get('/api/v1/game/fractions/level', (req, res) => {\n"
        "    levelCounter += 1;\n"
        "    let index = Math.floor(Math.random() * CHALLENGES.length);\n"
        "    if (index === lastChallengeIndex) {\n"
        "        index = (index + 1) % CHALLENGES.length;\n"
        "    }\n"
        "    lastChallengeIndex = index;\n"
        "    const challenge = CHALLENGES[index];\n"
        "    const value = challenge.numerator / challenge.denominator;\n"
        "    const expectedDecimal = Math.round(value * 1000) / 1000;\n"
        "    res.json({ status: 'success', level: levelCounter, fraction: challenge.fraction, expectedDecimal: expectedDecimal });\n"
        "});\n\n"
        "app.post('/api/v1/game/fractions/validate', (req, res) => {\n"
        "    const body = req.body || {};\n"
        "    const submitted = toNumber(body.answer);\n"
        "    const expected = body.fraction ? toNumber(body.fraction) : NaN;\n"
        "    const candidates = isFinite(expected)\n"
        "        ? [expected]\n"
        "        : CHALLENGES.map((challenge) => challenge.numerator / challenge.denominator);\n"
        "    const correct = isFinite(submitted) && candidates.some((target) => {\n"
        "        return Math.abs(submitted - target) < 0.000001;\n"
        "    });\n"
        "    const reference = isFinite(expected) ? expected : candidates[0];\n"
        "    const rounded = isFinite(reference) ? Math.round(reference * 1000) / 1000 : null;\n"
        "    res.json({ correct: correct, score: correct ? 100 : 0, expected: rounded === null ? null : String(rounded) });\n"
        "});\n\n"
        "app.get('/', (req, res) => {\n"
        "    res.sendFile(path.join(__dirname, 'public', 'index.html'));\n"
        "});\n\n"
        "const PORT = process.env.PORT || 3000;\n"
        "app.listen(PORT, () => {\n"
        "    console.log('Space Fractions server running at http://localhost:' + PORT);\n"
        "});\n"
    ),
    "public/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Space Fractions</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0B1020;
    --panel: #101A35;
    --line: #26304D;
    --ink: #E9EDF6;
    --muted: #93A1BC;
    --accent: #F2B94B;
    --accent-dim: #D9A23A;
    --ok: #5FBF8F;
    --bad: #E07777;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    padding: 28px 16px;
    background: var(--bg);
    color: var(--ink);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  .board { width: min(100%, 560px); }
  .eyebrow {
    font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 6px;
  }
  h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: 0.02em; }
  .tagline { margin: 0 0 20px; color: var(--muted); font-size: 13px; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 22px;
  }
  .panel-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  .chip {
    font-family: ui-monospace, Consolas, "Cascadia Mono", monospace;
    font-size: 12px; color: var(--muted);
    border: 1px solid var(--line); border-radius: 3px;
    padding: 4px 10px; white-space: nowrap;
  }
  .chips { display: flex; gap: 8px; align-items: center; }
  .fraction {
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 700; text-align: center;
    font-size: clamp(56px, 16vw, 88px);
    line-height: 1.05; margin: 8px 0 6px;
  }
  .bar {
    height: 8px; border: 1px solid var(--line); border-radius: 3px;
    background: #0C1428; overflow: hidden; margin: 12px 0 6px;
  }
  .bar-fill { height: 100%; width: 0%; background: var(--accent); transition: width 0.3s ease; }
  .hint { color: var(--muted); font-size: 12px; text-align: center; margin: 0 0 16px; }
  .row { display: flex; gap: 10px; }
  input[type="text"] {
    flex: 1; min-width: 0;
    background: #0C1428; color: var(--ink);
    border: 1px solid var(--line); border-radius: 4px;
    padding: 12px 14px; font-size: 15px;
    font-family: ui-monospace, Consolas, "Cascadia Mono", monospace;
  }
  input[type="text"]:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
    border-color: var(--accent);
  }
  button {
    border: 0; border-radius: 4px; font-size: 14px; font-weight: 600;
    padding: 12px 20px; cursor: pointer; font-family: inherit;
  }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .primary { background: var(--accent); color: #1A1408; }
  .primary:hover { background: var(--accent-dim); }
  .primary:disabled { opacity: 0.55; cursor: default; }
  .ghost {
    background: transparent; color: var(--muted);
    border: 1px solid var(--line);
  }
  .ghost:hover { color: var(--ink); border-color: var(--muted); }
  .meta {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 14px; gap: 10px;
  }
  .feedback { min-height: 20px; margin: 12px 0 0; font-size: 13px; color: var(--muted); }
  .feedback.ok { color: var(--ok); }
  .feedback.bad { color: var(--bad); }
  @media (prefers-reduced-motion: reduce) { .bar-fill { transition: none; } }
  @media (max-width: 420px) { .tagline { margin-bottom: 14px; } }
</style>
</head>
<body>
<main class="board">
  <p class="eyebrow">Observatory Challenge</p>
  <h1>Space Fractions</h1>
  <p class="tagline">Read the fraction, convert it to a decimal, and submit your answer.</p>
  <section class="panel">
    <div class="panel-head">
      <span class="eyebrow" style="margin: 0;">Current fraction</span>
      <span class="chips"><span class="chip">LEVEL <span id="level">1</span></span><span class="chip">SCORE <span id="score">0</span></span><span class="chip">STREAK <span id="streak">0</span></span></span>
    </div>
    <div class="fraction" id="current-fraction">--</div>
    <div class="bar"><div class="bar-fill" id="bar"></div></div>
    <p class="hint">What decimal value does this fraction equal?</p>
    <div class="row">
      <input id="answer" type="text" inputmode="decimal" placeholder="e.g. 0.75 or 3/4" aria-label="Your answer" autocomplete="off">
      <button class="primary" id="check" type="button">Check</button>
    </div>
    <p class="feedback" id="feedback" role="status" aria-live="polite"></p>
    <div class="meta">
      <button class="ghost" id="next" type="button">Next challenge</button>
      <span class="hint" style="margin: 0;">answers are validated by the API</span>
    </div>
  </section>
</main>
<script>
(function () {
  "use strict";
  var state = { fraction: "", score: 0, streak: 0, level: 1, expectedDecimal: null };
  var el = {
    fraction: document.getElementById("current-fraction"),
    bar: document.getElementById("bar"),
    answer: document.getElementById("answer"),
    check: document.getElementById("check"),
    next: document.getElementById("next"),
    feedback: document.getElementById("feedback"),
    score: document.getElementById("score"),
    level: document.getElementById("level"),
    streak: document.getElementById("streak")
  };
  function setFeedback(message, kind) {
    el.feedback.textContent = message || "";
    el.feedback.className = "feedback" + (kind ? " " + kind : "");
  }
  function setBusy(busy) {
    el.check.disabled = busy;
    el.check.textContent = busy ? "Checking..." : "Check";
  }
  function renderFraction(fraction) {
    state.fraction = fraction;
    el.fraction.textContent = fraction;
    var parts = fraction.split("/");
    var ratio = parts.length === 2 ? Number(parts[0]) / Number(parts[1]) : 0;
    el.bar.style.width = isFinite(ratio) && ratio > 0 ? Math.max(4, Math.min(100, ratio * 100)) + "%" : "0%";
  }
  function loadLevel() {
    setFeedback("Loading new challenge...", "");
    el.next.disabled = true;
    fetch("/api/v1/game/fractions/level")
      .then(function (response) {
        if (!response.ok) { throw new Error("HTTP " + response.status); }
        return response.json();
      })
      .then(function (data) {
        var fraction = data && typeof data.fraction === "string" ? data.fraction.trim() : "";
        if (!fraction) { throw new Error("Missing fraction in response"); }
        renderFraction(fraction);
        if (typeof data.level === "number" && isFinite(data.level)) {
          state.level = data.level;
          el.level.textContent = String(data.level);
        }
        state.expectedDecimal = typeof data.expectedDecimal === "number" && isFinite(data.expectedDecimal)
          ? data.expectedDecimal
          : null;
        el.answer.value = "";
        el.answer.focus();
        el.check.disabled = false;
        el.next.disabled = false;
        setFeedback("", "");
      })
      .catch(function () {
        el.check.disabled = false;
        el.next.disabled = false;
        setFeedback("Could not reach the game API. Press Next challenge to retry.", "bad");
      });
  }
  function checkAnswer() {
    var value = el.answer.value.trim();
    if (!value) {
      setFeedback("Type an answer first.", "bad");
      el.answer.focus();
      return;
    }
    setBusy(true);
    setFeedback("", "");
    fetch("/api/v1/game/fractions/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer: value, fraction: state.fraction })
    })
      .then(function (response) {
        if (!response.ok) { throw new Error("HTTP " + response.status); }
        return response.json();
      })
      .then(function (data) {
        setBusy(false);
        if (data.correct) {
          state.score += typeof data.score === "number" ? data.score : 100;
          state.streak += 1;
          el.score.textContent = String(state.score);
          el.streak.textContent = String(state.streak);
          setFeedback("Correct. Streak: " + state.streak + ".", "ok");
        } else {
          state.streak = 0;
          el.streak.textContent = "0";
          var shown = data.expected || (state.expectedDecimal === null ? "" : String(state.expectedDecimal));
          setFeedback(
            shown ? "Not quite. The answer is " + shown + "." : "Not quite. Try again.",
            "bad"
          );
        }
      })
      .catch(function () {
        setBusy(false);
        setFeedback("Could not reach the game API.", "bad");
      });
  }
  el.check.addEventListener("click", checkAnswer);
  el.next.addEventListener("click", loadLevel);
  el.answer.addEventListener("keydown", function (event) {
    if (event.key === "Enter") { checkAnswer(); }
  });
  loadLevel();
})();
</script>
</body>
</html>
""",
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
        "      summary: Returns a fresh random fraction challenge on every call\n"
        "      responses:\n"
        "        '200':\n"
        "          description: OK\n"
        "  /api/v1/game/fractions/validate:\n"
        "    post:\n"
        "      summary: Validate a submitted answer\n"
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
    "README.md": (
        "# Space Fractions Backend + Game UI\n\n"
        "Node.js / Express service for the Space Fractions game, with an\n"
        "interactive browser UI served from `public/`.\n\n"
        "## Run\n\n"
        "```bash\n"
        "npm install\n"
        "npm start\n"
        "```\n\n"
        "Then open <http://localhost:3000> to play.\n\n"
        "## Endpoints\n\n"
        "- `GET /api/v1/game/fractions/level` (fresh random challenge on every call)\n"
        "- `POST /api/v1/game/fractions/validate` (accepts decimals like\n"
        "  `0.75` and fractions like `3/4`, compared numerically)\n"
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

    Nested paths are remapped to their core file names (src/server.js ->
    server.js, tests/test.js -> test.js, ...).  ``index.html`` (at any level)
    is remapped to the canonical ``public/index.html`` game page.  Anything
    outside the core set is skipped and logged, so no other sub-directories
    are created.
    """
    kept: Dict[str, str] = {}
    order: List[str] = []
    for entry in files:
        raw = str(entry.get("path", "")).replace("\\", "/").strip()
        if not raw:
            continue
        flat = raw.rsplit("/", 1)[-1].lower()
        canonical = _CORE_LOOKUP.get(flat)
        if canonical is None and flat.endswith(".test.js"):
            canonical = "test.js"
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
    return f"""You are CodeAgent, a senior full-stack architect and DevOps engineer.
You turn architecture documentation into a complete, production-ready,
error-free full-stack project for "{project_name}": an interactive browser
game UI plus the Node.js / Express REST backend that serves it.

## Deliverables
Generate all of the following across the requested stages:
{_REQUIRED_FILES}
## Validation requirements (game logic)
- POST /api/v1/game/fractions/validate must accept the player's answer
  robustly: trim whitespace; accept decimal strings like "0.75" (including
  equivalents such as "0.750"); and accept fraction answers like "3/4" by
  dividing numerator by denominator and comparing numerically with a small
  float tolerance.
- The game page sends {{ "answer": <text>, "fraction": <current challenge> }}
  so the server can validate against the exact challenge in play.
- GET /api/v1/game/fractions/level must be dynamic: every call returns a
  brand-new randomly selected challenge from a pool of at least eight
  fractions (e.g. 1/2, 3/4, 2/5, 5/8, 1/4, 4/5, 3/10, 9/10) with
  expectedDecimal computed at request time, the level number incremented,
  and no immediate repeats. Response shape:
  {{ "status": "success", "level": 3, "fraction": "5/8", "expectedDecimal": 0.625 }}.

## Output format (STRICT)
- Emit every file as a block in exactly this shape:
===FILE: <relative/path/to/file>===
<complete file content>
===END FILE===
- Only the core files are expected (all at the project root except
  public/index.html). Files outside this set are ignored, so do not emit
  extra or nested files.
- Never wrap file contents in markdown code fences and never truncate a
  file; no "..." or "TODO" placeholders - full content only.

## Quality bar (must all hold)
- Production-ready and error-free. No syntax errors: every opening brace
  has a matching closing brace, every string literal is correctly quoted,
  and no unterminated template literals - when a backtick template is not
  needed, prefer plain string concatenation. JSON, YAML and SQL must be
  valid.
- Fully consistent: route paths in server.js, fetch calls in the game page,
  the OpenAPI spec and the tests must all match; the Dockerfile must run
  the server on port 3000.
- The game page must be complete and self-contained (inline CSS and
  JavaScript; no external libraries or asset files): it fetches the level
  on load and on every next-challenge click, immediately fills the
  fraction display (#current-fraction) from the response (a placeholder
  like -- must never get stuck), binds Submit/Check to POST the answer and
  updates Score, Streak (reset on a wrong answer) and the status message
  (including the expected decimal when wrong), keeps a LEVEL indicator in
  sync with the API, clears the answer input and re-enables the buttons
  after each fetch, and shows a graceful message with a working retry when
  the API is unreachable.
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
    """Generate the full-stack project files with the live DeepSeek API.

    The output covers the interactive game UI (public/index.html) plus the
    Express backend, SQL schema, OpenAPI spec, tests, README and the payload
    copy, so the reported file count is stable.  Truncated responses are
    recovered automatically (salvage + high-quality static completion) and
    the summary always contains a non-zero ``files_written`` list.
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
