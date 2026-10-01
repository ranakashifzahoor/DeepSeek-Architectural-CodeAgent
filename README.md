# DeepSeek Architectural Code Agent

A **Windows desktop application** that turns architectural documentation into a complete, runnable codebase using the official **DeepSeek API**.

The agent reads two design documents — `Architecture_Documentation.md` (prose) and `Architecture_View.md` (PlantUML views) — cleans and structures them into a single JSON payload, then generates the full **Space Fractions** backend project — `package.json`, `server.js`, `Dockerfile`, `schema.sql`, `openapi.yaml`, `test.js` and the structured payload — writing every file directly into a selectable output folder. Generation uses the built-in template engine by default, with optional live DeepSeek API generation.

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
- **Two generation modes** — the default built-in template generator writes the 7 project files directly (no network, fully deterministic); optionally, live generation sends the parsed architecture to the official DeepSeek API (`/v1/chat/completions`) in four focused stages.
- **Modern dark-mode GUI** — CustomTkinter dashboard with masked API key (show/hide toggle), file and folder pickers, live execution log and staged progress bar.
- **Background execution** — generation runs in a background thread; the window never freezes.
- **Safety built in** — path-traversal protection and response-truncation detection for API generations; the built-in mode guarantees output even with no API connectivity.
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
                                         agent_core.py
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                   built-in templates                 DeepSeek API (optional)
                   (default mode, 7 files)            4 staged generation calls
                              │                               │
                              └───────────────┬───────────────┘
                                              ▼
                                     output/  (generated project files)
```

### Modules

| File | Responsibility | Highlights |
| --- | --- | --- |
| `parser.py` | Reads both architecture documents, cleans prose/PlantUML, structures everything into a JSON payload | `build_payload()`, `clean_plantuml()`, `extract_sections()`; fence-aware heading detection; standalone CLI included |
| `agent_core.py` | Generation engine with two modes: built-in template generator (default, direct 7-file output) and live DeepSeek API generation (optional) | `write_builtin_project_files()`, `DeepSeekClient` (retries, backoff), 4-stage prompting, strict `===FILE: <path>=== … ===END FILE===` protocol, path-traversal guard, truncation detection |
| `main_gui.py` | CustomTkinter desktop dashboard; orchestrates parse → generate in a background thread | Masked API-key entry, file/folder pickers, queue-driven live log, staged progress, success/fallback/error popups |
| `build_exe.py` | One-command PyInstaller build for the standalone executable | `--onefile --windowed`, bundles CustomTkinter assets, `parser`/`agent_core`, Tcl/Tk runtime hooks |

### How the file protocol works

In live API mode, DeepSeek is instructed to answer with one block per file:

```
===FILE: src/server.js===
<complete file content>
===END FILE===
```

The engine validates the block structure (detecting truncated responses), de-fences any stray markdown, refuses absolute or escaping paths, and writes the files into the chosen output folder. A JSON file-mapping response is also accepted as a fallback for robustness.

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
2. **Enter your DeepSeek API key** *(only needed for live API generation)* — the field is masked; use the **Show** checkbox to reveal it. The key is only kept in memory for the session and is never written to disk.
3. **Select the Architecture Documentation** — click **Browse…** next to *Architecture Documentation* and choose `Architecture_Documentation.md`.
4. **Select the Architecture Views** — choose `Architecture_View.md` (PlantUML diagrams).
5. **Choose the target output directory** — any folder; it is created automatically if missing.
6. **Click Generate Project Code** — watch the live log and the progress bar. By default, the built-in template generator writes the 7 project files (`package.json`, `server.js`, `Dockerfile`, `schema.sql`, `openapi.yaml`, `test.js`, `architecture_payload.json`) instantly; tick **Use DeepSeek API for live generation** to generate the project with the DeepSeek model instead (API key required).
7. **Review the results** — when the success popup appears ("Generated 7 file(s)"), your output folder contains the complete project. In live-API mode, failures raise an error popup with the exact reason instead of silently writing placeholders.

**Batch/CLI usage (optional)**

```bash
# parse only, inspect the JSON payload
python parser.py --doc Architecture_Documentation.md --view Architecture_View.md --out architecture_payload.json
```

---

## Generated Project Architecture

The agent produces a typical Node.js/Express service layout (exact file names may vary slightly as the code is model-generated; the required deliverables are enforced by the stage prompts):

```
output/
├── package.json               # npm manifest (start / test scripts, pinned deps)
├── src/                       # Express backend
│   ├── server.js              # HTTP entry point
│   ├── app.js                 # application wiring / middleware
│   ├── routes/                # API route definitions
│   ├── controllers/           # request handlers
│   └── services/              # business logic + data access
├── sql/
│   └── schema.sql             # SQL DDL — tables, keys, constraints, indexes
├── openapi.yaml               # OpenAPI 3.0 specification for every endpoint
├── tests/                     # automated test suite (Jest + Supertest)
├── Dockerfile                 # container image for the backend
├── .dockerignore
├── .env.example               # configuration template
├── README.md                  # README for the generated project
└── architecture_payload.json  # parsed architecture input (traceability)
```

**Note:** the SQL schema matches the documented data model; Express routes, OpenAPI paths and the tests are generated consistently with each other.

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
| GUI title, accent color, window size | `main_gui.py` — `APP_TITLE`, `ACCENT`, `geometry()` |
| PyInstaller flags / executable name | `build_exe.py` |
| Ignored files & folders | `.gitignore` |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Missing API key` warning | Enter your DeepSeek API key in the GUI (or set the `DEEPSEEK_API_KEY` environment variable for source runs). |
| API error popup (live mode) | The log shows the exact HTTP status and message. Check key validity, internet connection and API credits — or use the default built-in mode, which needs no API. |
| Wrong model name error | Adjust `DEFAULT_MODEL` in `agent_core.py` to a model available to your DeepSeek account. |
| `ModuleNotFoundError: tkinter` (source) | Use a full CPython build that includes Tcl/Tk (the python.org installer does by default). |
| Antivirus flags the one-file exe | Common for freshly built PyInstaller binaries — allow it, or rebuild with `--console` for inspection. |
| First launch takes a few seconds | Normal: the one-file executable unpacks itself to a temp folder. |

---

## Notes

- The default built-in generation works fully offline; live DeepSeek generation requires a valid API key (usage is billed by DeepSeek according to their pricing).
- The API key is never stored: it lives only in the application's memory for the current session (`.env` files are git-ignored for source runs).
- Generated code is produced by an LLM — review it before using it in production.
