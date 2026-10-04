# DeepSeek Architectural Code Agent

A **Windows desktop application** that turns architectural documentation into a complete, runnable codebase using the official **DeepSeek API**.

The agent reads two design documents — `Architecture_Documentation.md` (prose) and `Architecture_View.md` (PlantUML views) — cleans and structures them into a single JSON payload, then generates the full **Space Fractions** project with the **live DeepSeek API**: a Node.js/Express backend, `package.json`, SQL schema, OpenAPI 3.0 specification, Dockerfile and automated tests — all written directly into a selectable output folder.

Built with Python + CustomTkinter and shipped as a single-file executable (`DeepSeek_CodeAgent.exe`) that requires **no Python runtime** on target machines.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture & Design](#architecture--design)
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage Guide](#usage-guide)
- [Generated Project Architecture](#generated-project-architecture)
- [Building the Standalone Executable](#building-the-standalone-executable)
- [Repository Structure](#repository-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

---

## Project Overview

This tool is a simplified **Code Agent** (in the spirit of Claude Code–style agents) specialized for generating a complete backend project from its architecture documents.

**Key features**

- **Document parser** — extracts heading-aware sections and cleans PlantUML diagrams from the two markdown inputs into a structured JSON payload.
- **DeepSeek code generation** — sends the parsed architecture to the official DeepSeek API (`/v1/chat/completions`) in five focused stages and writes the full-stack core files (Express backend + interactive game UI, 8 files + payload) in a strict, machine-parsable format.
- **Polished modern UI** — CustomTkinter dashboard with an emerald-and-white theme, masked API key (show/hide toggle), file and folder pickers, colored execution log and staged progress bar.
- **Background execution** — generation runs in a background thread; the window never freezes.
- **Safety built in** — path-traversal protection, response-truncation recovery (incomplete API responses are salvaged or completed with fallback files) and automatic retries with backoff.
- **Single-file executable** — PyInstaller bundles the app, all dependencies, CustomTkinter themes/fonts and Tcl/Tk into one `DeepSeek_CodeAgent.exe`.

---

## Architecture & Design

### Pipeline

```
Architecture_Documentation.md ─┐
                               ├─► parser.py ──► architecture_payload.json
Architecture_View.md ──────────┘        (cleaned text + sections + PlantUML)
                                              │
                                              ▼
                                    agent_core.py  ⇄  DeepSeek API
                                        stage 1: package.json + server.js
                                        stage 2: public/index.html (game UI)
                                        stage 3: schema.sql
                                        stage 4: openapi.yaml + README.md
                                        stage 5: test.js + Dockerfile
                                              │
                                              ▼
                                     output/  (generated project files)
```

### Modules

| File | Responsibility | Highlights |
| --- | --- | --- |
| `parser.py` | Reads both architecture documents, cleans prose/PlantUML, structures everything into a JSON payload | `build_payload()`, `clean_plantuml()`, `extract_sections()`; fence-aware heading detection; standalone CLI included |
| `agent_core.py` | DeepSeek API client + staged generation engine; writes the full-stack core project files (Express backend + interactive game UI) into the output directory | `DeepSeekClient` (retries, backoff), 5-stage prompting, strict `===FILE: <path>=== … ===END FILE===` protocol, core-file scope (8 files + payload; game UI in public/), path-traversal guard, truncation recovery with high-quality fallback |
| `main_gui.py` | CustomTkinter desktop dashboard (modern emerald-and-white theme); orchestrates parse → generate in a background thread | Masked API-key entry, file/folder pickers, queue-driven colored log panel, staged progress, success and error popups |
| `build_exe.py` | One-command PyInstaller build for the standalone executable | `--onefile --windowed`, bundles CustomTkinter assets, `parser`/`agent_core`, Tcl/Tk runtime hooks |

### How the file protocol works

DeepSeek is instructed to answer with one block per file:

```
===FILE: src/server.js===
<complete file content>
===END FILE===
```

The engine validates the block structure (truncated responses are salvaged by closing open file streams, and missing core files are completed automatically), de-fences any stray markdown, refuses absolute or escaping paths, and writes the core files into the chosen output folder. Only the core set is kept (package.json, server.js, public/index.html, Dockerfile, schema.sql, openapi.yaml, test.js, README.md): nested paths are flattened to their core names (index.html to public/index.html) and extra files are skipped and logged, so the output stays at 8-9 files. A JSON file-mapping response is also accepted for robustness.

### PyInstaller compilation

`build_exe.py` runs PyInstaller with:

- `--onefile` — everything in a single `.exe`
- `--windowed` — no console window pops up
- `--collect-all customtkinter` — bundles themes, fonts and assets
- hidden imports for `parser.py` / `agent_core.py`
- built-in PyInstaller runtime hooks for Tcl/Tk, so the GUI works without any installed Python

---

## Prerequisites & Installation

### Requirements

| | |
| --- | --- |
| OS | Windows 10 / 11 (x64) |
| API key | DeepSeek API key — create one at <https://platform.deepseek.com> |
| Network | Internet access for the generation calls |
| Source runs | Python 3.10+ (3.13 recommended, must include `tkinter`) |

### Run from source

```bash
cd CodeAgent_DeepSeek
python -m pip install -r requirements.txt
python main_gui.py
```

### Run the standalone executable

**Option A — use a prebuilt binary** (if provided): double-click `dist\DeepSeek_CodeAgent.exe`.

**Option B — build it yourself** (the compiled binary is not committed to the repository):

```bash
python build_exe.py
```

The executable is created at `dist\DeepSeek_CodeAgent.exe` — copy it anywhere and double-click. No Python installation is required on the target machine. The first launch takes a few seconds while the one-file bundle unpacks.

---

## Usage Guide

1. **Launch the application** — `python main_gui.py` (source) or double-click `dist\DeepSeek_CodeAgent.exe`.
2. **Enter your DeepSeek API key** — the field is masked; use the **Show** checkbox to reveal it. The key is only kept in memory for the session and is never written to disk.
3. **Select the Architecture Documentation** — the field shows a short placeholder until you pick a file; click **Browse…** and choose `Architecture_Documentation.md`.
4. **Select the Architecture Views** — choose `Architecture_View.md` (PlantUML diagrams).
5. **Choose the target output directory** — any folder; it is created automatically if missing.
6. **Click Generate Project Code** — watch the live log and the progress bar:
   - *Parsing* — both documents are cleaned and structured into a JSON payload;
   - *Stages* — DeepSeek generates the backend, SQL schema, documentation and tests in four API calls.
7. **Review the results** — when the success popup appears, your output folder contains the complete generated project. Run it with `npm install` + `npm start`, then open http://localhost:3000 to play the Space Fractions game. Failures show an error popup with the exact reason.

**Batch/CLI usage (optional)**

```bash
# parse only, inspect the JSON payload
python parser.py --doc Architecture_Documentation.md --view Architecture_View.md --out architecture_payload.json
```

---

## Generated Project Architecture

The agent is scoped to the essential core file set, so the output stays compact and predictable:

```
output/
├── package.json               # npm manifest (start / test scripts, pinned deps)
├── server.js                  # single-file Express backend (serves public/ + REST API)
├── public/
│   └── index.html             # interactive game UI (inline CSS + JS)
├── Dockerfile                 # container image for the backend
├── schema.sql                 # SQL DDL: tables, keys, constraints
├── openapi.yaml               # OpenAPI 3.0 specification for every endpoint
├── test.js                    # automated test suite (Jest + Supertest)
├── README.md                  # README for the generated project
└── architecture_payload.json  # parsed architecture input (traceability)
```

**Note:** the SQL schema matches the documented data model; the game page, Express routes, OpenAPI paths and the tests are generated consistently with each other. The file count stays at 8-9 files (core set + payload copy). Run the generated project with `npm install` + `npm start` and open http://localhost:3000 to play; answers like `0.75` and `3/4` are both accepted.

---

## Building the Standalone Executable

```bash
python build_exe.py             # windowed one-file build (normal use)
python build_exe.py --console   # keep a console window (debugging)
python build_exe.py --check     # print the PyInstaller command only
```

Output: `dist\DeepSeek_CodeAgent.exe` (~14 MB).

Tips:

- If distributing the app, attach the built `.exe` to a **GitHub Release** rather than committing it to the repository.
- An application icon can be added with PyInstaller's `--icon file.ico` option.

---

## Repository Structure

```
CodeAgent_DeepSeek/
├── main_gui.py            # Desktop dashboard (CustomTkinter)
├── parser.py              # Architecture document → JSON payload
├── agent_core.py          # DeepSeek generation engine
├── build_exe.py           # One-command PyInstaller build
├── requirements.txt       # Python dependencies
├── README.md
├── .gitignore
├── build/                 # generated during build (ignored)
└── dist/                  # generated executable output (ignored)
```

---

## Customization

| What | Where |
| --- | --- |
| Model name (`deepseek-chat`), timeout, retries | `agent_core.py` — `DEFAULT_MODEL`, `DEFAULT_TIMEOUT`, `DEFAULT_MAX_RETRIES` |
| Generation stages / prompts | `agent_core.py` — `DEFAULT_STAGES`, `build_system_prompt()` |
| GUI title, theme colors, window size | `main_gui.py` — `APP_TITLE`, design tokens in class `T`, `geometry()` |
| PyInstaller flags / executable name | `build_exe.py` |
| Ignored files & folders | `.gitignore` |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Missing API key` warning | Enter your DeepSeek API key in the GUI (or set the `DEEPSEEK_API_KEY` environment variable for source runs). |
| API error popup | The log shows the exact HTTP status and message. Check key validity, internet connection and API credits, then retry. |
| Incomplete / truncated API response | Handled automatically: the agent salvages partial files and completes the missing set with fallback files; details appear in the log. |
| Wrong model name error | Adjust `DEFAULT_MODEL` in `agent_core.py` to a model available to your DeepSeek account. |
| `ModuleNotFoundError: tkinter` (source) | Use a full CPython build that includes Tcl/Tk (the python.org installer does by default). |
| Antivirus flags the one-file exe | Common for freshly built PyInstaller binaries — allow it, or rebuild with `--console` for inspection. |
| First launch takes a few seconds | Normal: the one-file executable unpacks itself to a temp folder. |

---

## Notes

- Generation requires a valid DeepSeek API key; usage is billed by DeepSeek according to their pricing.
- The API key is never stored: it lives only in the application's memory for the current session (`.env` files are git-ignored for source runs).
- Generated code is produced by an LLM — review it before using it in production.
