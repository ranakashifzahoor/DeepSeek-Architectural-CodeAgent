"""
main_gui.py - CustomTkinter dashboard for the CodeAgent_DeepSeek pipeline.

A modern "Emerald & White" desktop dashboard that:

    1. collects the DeepSeek API key, the two architecture documents and an
       output directory,
    2. parses the documents with ``parser.py``,
    3. runs the live DeepSeek generation (``agent_core.py``) in a background
       thread so the window never freezes,
    4. streams live status / terminal output into the log panel, and
    5. shows a clean success popup with the exact file count when a run
       finishes (truncated API responses are recovered automatically by the
       engine, so the count is never zero).

Generation is performed live by the official DeepSeek API.  A valid
DeepSeek API key is required.

Run with::

    python main_gui.py

Optional automation helper::

    python main_gui.py --smoke-test      # opens the window, closes it after ~4 s
"""

from __future__ import annotations

import argparse
import queue
import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Optional

import customtkinter as ctk

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# When packaged with PyInstaller, ``__file__`` points inside the temporary
# one-file extraction dir - default paths should then sit next to the .exe.
BASE_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else SCRIPT_DIR
)

import agent_core  # noqa: E402  (imported after sys.path setup)
import parser as arch_parser  # noqa: E402  (imported after sys.path setup)

__all__ = ["CodeAgentApp", "main"]

APP_TITLE = "DeepSeek Architectural Code Agent"
GENERATE_LABEL = "Generate Project Code"
GENERATE_RUNNING_LABEL = "Generating..."


# --------------------------------------------------------------------------- #
# Design tokens - "Emerald Green & Crisp White" modern dashboard
# --------------------------------------------------------------------------- #

class T:
    BG = "#F8FAF9"            # page background (soft off-white)
    SURFACE = "#FFFFFF"       # cards
    SURFACE_2 = "#FFFFFF"     # inputs
    HOVER = "#F1F5F9"         # hover state
    BORDER = "#E2E8F0"        # subtle 1px structure
    BORDER_STRONG = "#CBD5E1"
    TEXT = "#0F172A"          # headers / primary text (deep slate)
    TEXT_DIM = "#475569"      # secondary text
    TEXT_FAINT = "#94A3B8"    # placeholders / muted meta
    ACCENT = "#10B981"        # emerald (primary buttons / accents)
    ACCENT_HOVER = "#059669"
    ACCENT_TEXT = "#059669"   # darker emerald for text on light backgrounds
    SUCCESS = "#047857"       # success badge text
    SUCCESS_BG = "#D1FAE5"    # success badge background
    RUN_BG = "#ECFDF5"
    RUN_TX = "#059669"
    WARN_BG = "#FEF3C7"
    WARN_TX = "#92400E"
    ERROR = "#B91C1C"         # error badge text
    ERROR_BG = "#FEE2E2"
    IDLE_BG = "#F1F5F9"
    IDLE_TX = "#64748B"
    LOG_BG = "#F8FAFC"
    LOG_TEXT = "#334155"
    LOG_DIM = "#94A3B8"
    LOG_SUCCESS = "#047857"
    LOG_ERROR = "#B91C1C"

    UI_FAMILY = "Segoe UI"


def _ui_font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=T.UI_FAMILY, size=size, weight=weight)


def _pick_ui_family() -> str:
    """Prefer Inter when installed, otherwise the native Segoe UI."""
    try:
        from tkinter import font as tkfont

        available = set(tkfont.families())
        for name in ("Inter", "Segoe UI"):
            if name in available:
                return name
    except Exception:
        pass
    return "Segoe UI"


_MONO_CANDIDATES = ("Cascadia Mono", "Cascadia Code", "Consolas")


def _pick_mono_family() -> str:
    """Pick the best available monospace family for the log panel."""
    try:
        from tkinter import font as tkfont

        available = set(tkfont.families())
        for name in _MONO_CANDIDATES:
            if name in available:
                return name
    except Exception:
        pass
    return "Consolas"


class CodeAgentApp(ctk.CTk):
    """Main window of the desktop dashboard."""

    def __init__(self) -> None:
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")
        super().__init__(fg_color=T.BG)

        self.title(APP_TITLE)
        # Window size tuned to fit 720p-class workspaces; CustomTkinter
        # multiplies these values by the display DPI factor on Windows.
        self.geometry("1000x600+140+30")
        self.minsize(900, 580)

        self._font_family = _pick_ui_family()
        self._mono_family = _pick_mono_family()
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._stage_count = 0
        self._parse_done = False
        self._total_steps = 1 + len(agent_core.DEFAULT_STAGES)
        self._current_out_dir = ""
        self._doc_path = BASE_DIR / arch_parser.DOC_FILE_NAME
        self._view_path = BASE_DIR / arch_parser.VIEW_FILE_NAME
        self._out_path = BASE_DIR / "output"

        self._build_ui()
        self.after(80, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._append_log(
            "[gui] Ready. Select the two architecture documents and an output "
            'folder, then click "Generate Project Code".',
            tag="dim",
        )
        self._append_log(
            "[gui] Generation runs live on the DeepSeek API "
            "(4 stages: backend, database, docs, tests).",
            tag="dim",
        )

    # ------------------------------------------------------------------ UI --

    def _ui(self, size: int = 13, weight: str = "normal") -> ctk.CTkFont:
        return ctk.CTkFont(family=self._font_family, size=size, weight=weight)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header ------------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 4))
        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(anchor="w")
        ctk.CTkLabel(
            title_row,
            text="DeepSeek",
            font=self._ui(23, "bold"),
            text_color=T.ACCENT_TEXT,
        ).pack(side="left")
        ctk.CTkLabel(
            title_row,
            text=" Architectural Code Agent",
            font=self._ui(23, "bold"),
            text_color=T.TEXT,
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text=(
                "Generate a complete Node.js / Express backend from your "
                "architecture documents with the live DeepSeek API."
            ),
            font=self._ui(12),
            text_color=T.TEXT_DIM,
        ).pack(anchor="w", pady=(2, 0))

        # Configuration card -------------------------------------------------
        card = ctk.CTkFrame(
            self,
            fg_color=T.SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=T.BORDER,
        )
        card.grid(row=1, column=0, sticky="ew", padx=24, pady=(14, 10))
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            card, text="Configuration", font=self._ui(13, "bold"), text_color=T.TEXT
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=18, pady=(12, 6))

        self.api_key_entry = self._form_row(
            card, 1, "DeepSeek API key", placeholder="sk-..."
        )
        self.show_key_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            card,
            text="Show",
            variable=self.show_key_var,
            command=self._toggle_key_visibility,
            width=70,
            font=self._ui(12),
            fg_color=T.ACCENT,
            hover_color=T.ACCENT_HOVER,
            border_color=T.BORDER_STRONG,
            text_color=T.TEXT_DIM,
            checkbox_width=18,
            checkbox_height=18,
            corner_radius=5,
        ).grid(row=1, column=2, padx=(0, 18), pady=(0, 6), sticky="w")

        self.doc_entry = self._form_row(
            card,
            2,
            "Documentation",
            placeholder="Architecture_Documentation.md",
            browse_command=self._pick_documentation,
        )
        self.view_entry = self._form_row(
            card,
            3,
            "Architecture views",
            placeholder="Architecture_View.md",
            browse_command=self._pick_view,
        )
        self.out_entry = self._form_row(
            card,
            4,
            "Output folder",
            placeholder="Output Destination",
            browse_command=self._pick_output_dir,
            bottom_pady=(0, 14),
        )

        # Actions --------------------------------------------------------------
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 8))
        actions.grid_columnconfigure(2, weight=1)

        self.generate_button = ctk.CTkButton(
            actions,
            text=GENERATE_LABEL,
            height=46,
            width=240,
            corner_radius=10,
            font=self._ui(14, "bold"),
            fg_color=T.ACCENT,
            hover_color=T.ACCENT_HOVER,
            text_color="#FFFFFF",
            command=self._on_generate,
        )
        self.generate_button.grid(row=0, column=0, padx=(0, 10))

        self.clear_button = ctk.CTkButton(
            actions,
            text="Clear log",
            width=96,
            height=46,
            corner_radius=10,
            font=self._ui(12),
            fg_color=T.SURFACE,
            hover_color=T.HOVER,
            border_width=1,
            border_color=T.BORDER,
            text_color=T.TEXT_DIM,
            command=self._clear_log,
        )
        self.clear_button.grid(row=0, column=1, padx=(0, 18))

        self.progress = ctk.CTkProgressBar(
            actions,
            height=8,
            corner_radius=4,
            fg_color=T.BORDER,
            progress_color=T.ACCENT,
        )
        self.progress.grid(row=0, column=2, sticky="ew")
        self.progress.set(0.0)

        # Log panel --------------------------------------------------------------
        log_card = ctk.CTkFrame(
            self,
            fg_color=T.SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=T.BORDER,
        )
        log_card.grid(row=3, column=0, sticky="nsew", padx=24, pady=(0, 8))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_card, text="Execution Log", font=self._ui(13, "bold"), text_color=T.TEXT
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(12, 0))

        self.log_box = ctk.CTkTextbox(
            log_card,
            wrap="word",
            font=ctk.CTkFont(family=self._mono_family, size=12),
            fg_color=T.LOG_BG,
            text_color=T.LOG_TEXT,
            corner_radius=8,
            border_width=1,
            border_color=T.BORDER,
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(8, 16))
        self.log_box.configure(state="disabled")

        # colored log tags (best effort - private widget access, guarded)
        self._text_widget = None
        try:
            self._text_widget = self.log_box._textbox  # noqa: SLF001
            self._text_widget.tag_configure("dim", foreground=T.LOG_DIM)
            self._text_widget.tag_configure("success", foreground=T.LOG_SUCCESS)
            self._text_widget.tag_configure("error", foreground=T.LOG_ERROR)
        except Exception:
            self._text_widget = None

        # Status badge ---------------------------------------------------------
        status_row = ctk.CTkFrame(self, fg_color="transparent")
        status_row.grid(row=4, column=0, sticky="w", padx=26, pady=(0, 12))
        self.status_badge = ctk.CTkFrame(status_row, fg_color=T.IDLE_BG, corner_radius=10)
        self.status_badge.pack(side="left")
        self.status_label = ctk.CTkLabel(
            self.status_badge,
            text="Ready",
            font=self._ui(11, "bold"),
            text_color=T.IDLE_TX,
        )
        self.status_label.pack(padx=12, pady=4)

    def _form_row(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label_text: str,
        initial: str = "",
        browse_command=None,
        placeholder: str = "",
        bottom_pady=(0, 6),
    ) -> ctk.CTkEntry:
        ctk.CTkLabel(
            parent,
            text=label_text,
            font=self._ui(12),
            text_color=T.TEXT_DIM,
            anchor="w",
            width=150,
        ).grid(row=row, column=0, padx=(18, 10), pady=bottom_pady, sticky="w")

        entry = ctk.CTkEntry(
            parent,
            height=38,
            corner_radius=8,
            fg_color=T.SURFACE_2,
            border_color=T.BORDER,
            border_width=1,
            text_color=T.TEXT,
            placeholder_text_color=T.TEXT_FAINT,
            font=self._ui(12),
        )
        if initial:
            entry.insert(0, initial)
        if placeholder:
            entry.configure(placeholder_text=placeholder)
        entry.grid(row=row, column=1, padx=(0, 10), pady=bottom_pady, sticky="ew")
        self._bind_focus_ring(entry)

        if browse_command is not None:
            ctk.CTkButton(
                parent,
                text="Browse...",
                width=92,
                height=38,
                corner_radius=8,
                font=self._ui(12),
                fg_color=T.SURFACE,
                hover_color=T.HOVER,
                border_width=1,
                border_color=T.BORDER,
                text_color=T.TEXT,
                command=browse_command,
            ).grid(row=row, column=2, padx=(0, 18), pady=bottom_pady)
        return entry

    @staticmethod
    def _bind_focus_ring(entry: ctk.CTkEntry) -> None:
        """Highlight the input border with the accent color on focus."""
        def on_focus_in(_event=None):
            try:
                entry.configure(border_color=T.ACCENT)
            except Exception:
                pass

        def on_focus_out(_event=None):
            try:
                entry.configure(border_color=T.BORDER)
            except Exception:
                pass

        try:
            entry.bind("<FocusIn>", on_focus_in)
            entry.bind("<FocusOut>", on_focus_out)
        except Exception:
            pass

    # -------------------------------------------------------------- actions --

    def _set_status(self, text: str, kind: str = "idle") -> None:
        palettes = {
            "idle": (T.IDLE_BG, T.IDLE_TX),
            "run": (T.RUN_BG, T.RUN_TX),
            "ok": (T.SUCCESS_BG, T.SUCCESS),
            "warn": (T.WARN_BG, T.WARN_TX),
            "err": (T.ERROR_BG, T.ERROR),
        }
        bg, fg = palettes.get(kind, palettes["idle"])
        try:
            self.status_badge.configure(fg_color=bg)
            self.status_label.configure(text=text, text_color=fg)
        except Exception:
            self.status_label.configure(text=text)

    def _toggle_key_visibility(self) -> None:
        self.api_key_entry.configure(show="" if self.show_key_var.get() else "*")

    def _pick_documentation(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Architecture_Documentation.md",
            initialdir=str(BASE_DIR),
            initialfile=arch_parser.DOC_FILE_NAME,
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
        )
        if path:
            self._doc_path = Path(path)
            self._set_entry(self.doc_entry, path)

    def _pick_view(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Architecture_View.md",
            initialdir=str(BASE_DIR),
            initialfile=arch_parser.VIEW_FILE_NAME,
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
        )
        if path:
            self._view_path = Path(path)
            self._set_entry(self.view_entry, path)

    def _pick_output_dir(self) -> None:
        initial = self.out_entry.get().strip() or str(self._out_path)
        path = filedialog.askdirectory(
            parent=self, title="Select target output directory", initialdir=initial
        )
        if path:
            self._out_path = Path(path)
            self._set_entry(self.out_entry, path)

    @staticmethod
    def _set_entry(entry: ctk.CTkEntry, value: str) -> None:
        entry.delete(0, "end")
        entry.insert(0, value)

    # --------------------------------------------------------------- logging --

    def _append_log(self, text: str, tag: Optional[str] = None) -> None:
        self.log_box.configure(state="normal")
        try:
            if tag and self._text_widget is not None:
                self._text_widget.insert("end", text + "\n", tag)
            else:
                self.log_box.insert("end", text + "\n")
            try:
                self.log_box.see("end")
            except AttributeError:  # pragma: no cover - older customtkinter
                self.log_box._textbox.see("end")  # noqa: SLF001
        except Exception:
            pass
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    @staticmethod
    def _tag_for(line: str) -> Optional[str]:
        text = line.strip()
        if "SUCCESS" in text:
            return "success"
        if text.startswith("Traceback") or "ERROR" in text or "failed" in text.lower():
            return "error"
        if text.startswith("[gui]"):
            return "dim"
        return None

    # ------------------------------------------------------------ generation --

    def _on_generate(self) -> None:
        if self._running:
            return

        api_key = self.api_key_entry.get().strip()
        doc_text = self.doc_entry.get().strip()
        view_text = self.view_entry.get().strip()
        out_text = self.out_entry.get().strip()
        doc_path = Path(doc_text) if doc_text else self._doc_path
        view_path = Path(view_text) if view_text else self._view_path
        out_value = out_text if out_text else str(self._out_path)
        self._doc_path = doc_path
        self._view_path = view_path

        if not api_key:
            messagebox.showwarning(
                "Missing API key",
                "Please enter your DeepSeek API key first.",
                parent=self,
            )
            return
        if not doc_path.is_file():
            messagebox.showwarning(
                "Documentation not found",
                f"Architecture documentation file not found:\n{doc_path}",
                parent=self,
            )
            return
        if not view_path.is_file():
            messagebox.showwarning(
                "Architecture view not found",
                f"Architecture view file not found:\n{view_path}",
                parent=self,
            )
            return
        if not out_value:
            messagebox.showwarning(
                "Missing output directory",
                "Please choose a target output directory.",
                parent=self,
            )
            return

        out_dir = Path(out_value)
        self._out_path = out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "Invalid output directory",
                f"Could not create:\n{out_dir}\n\n{exc}",
                parent=self,
            )
            return

        self._running = True
        self._stage_count = 0
        self._parse_done = False
        self._current_out_dir = str(out_dir)
        self._total_steps = 1 + len(agent_core.DEFAULT_STAGES)
        self.progress.set(0.0)
        self.generate_button.configure(state="disabled", text=GENERATE_RUNNING_LABEL)
        self._set_status("Starting...", "run")

        stamp = datetime.now().strftime("%H:%M:%S")
        self._append_log("-" * 60, tag="dim")
        self._append_log(f"[gui] Run started at {stamp}", tag="dim")
        self._append_log(f"[gui] Documentation : {doc_path}", tag="dim")
        self._append_log(f"[gui] Architecture  : {view_path}", tag="dim")
        self._append_log(f"[gui] Output folder : {out_dir}", tag="dim")

        self._worker = threading.Thread(
            target=self._worker_run,
            args=(doc_path, view_path, out_dir, api_key),
            daemon=True,
        )
        self._worker.start()

    def _worker_run(
        self, doc_path: Path, view_path: Path, out_dir: Path, api_key: str
    ) -> None:
        """Background worker: parse documents, then run the DeepSeek generation."""
        q = self._queue
        try:
            q.put(("status", "Parsing architecture documents..."))
            q.put(("log", "[gui] Parsing architecture documents..."))
            payload = arch_parser.build_payload(doc_path, view_path)

            doc_info = payload["documents"]["documentation"]
            view_info = payload["documents"]["architecture_view"]
            q.put(
                (
                    "log",
                    f"[gui] Parsed {doc_info['section_count']} documentation "
                    f"section(s), {doc_info['plantuml_diagram_count']} diagram(s); "
                    f"{view_info['section_count']} view section(s), "
                    f"{view_info['plantuml_diagram_count']} diagram(s).",
                )
            )
            q.put(("parsed", True))

            q.put(("status", "Generating project files with DeepSeek..."))
            q.put(("log", "[gui] Generating project files with the DeepSeek API..."))
            summary = agent_core.generate_project(
                payload,
                out_dir,
                api_key=api_key,
                log=lambda message: q.put(("log", str(message))),
            )
            q.put(("done", summary))
        except Exception as e:
            q.put(("log", traceback.format_exc()))
            q.put(("error", f"{type(e).__name__}: {e}"))

    # ---------------------------------------------------------- queue drain --

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    text = str(payload)
                    self._append_log(text, tag=self._tag_for(text))
                    self._track_progress(text)
                elif kind == "progress":
                    self.progress.set(float(payload))
                elif kind == "status":
                    self._set_status(str(payload), "run")
                elif kind == "parsed":
                    self._parse_done = True
                    self._update_stage_progress()
                elif kind == "done":
                    self.progress.set(1.0)
                    self._finish_run()
                    used_fallback = False
                    if isinstance(payload, dict):
                        files = payload.get("files_written", [])
                        out_dir_str = payload.get("output_dir") or str(
                            getattr(self, "_current_out_dir", "")
                        )
                        used_fallback = bool(payload.get("used_fallback"))
                    else:
                        files = [str(p) for p in (payload or [])]
                        out_dir_str = str(getattr(self, "_current_out_dir", ""))
                    if used_fallback:
                        self._set_status(
                            f"Completed with recovery - {len(files)} file(s)", "warn"
                        )
                        messagebox.showinfo(
                            "Generation complete",
                            f"Generated {len(files)} file(s) into:\n{out_dir_str}\n\n"
                            "Note: the API response was incomplete, so recovery "
                            "files were added to complete the project. See the log "
                            "for details.",
                            parent=self,
                        )
                    else:
                        self._set_status(f"Completed - {len(files)} file(s)", "ok")
                        messagebox.showinfo(
                            "Generation complete",
                            f"Generated {len(files)} file(s) into:\n{out_dir_str}",
                            parent=self,
                        )
                elif kind == "error":
                    self._finish_run()
                    self._set_status("Failed. Check the log for details.", "err")
                    messagebox.showerror("Generation failed", str(payload), parent=self)
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

    def _track_progress(self, line: str) -> None:
        if ": requesting" in line:
            match = re.search(r"stage '([^']+)'", line)
            if match:
                self._set_status(
                    f"DeepSeek is generating: stage '{match.group(1)}'...", "run"
                )
        if "wrote" in line and "file(s)" in line:
            self._stage_count += 1
            self._update_stage_progress()

    def _update_stage_progress(self) -> None:
        done = (1 if self._parse_done else 0) + self._stage_count
        total = max(self._total_steps, 1)
        self.progress.set(min(done / total, 1.0))

    def _finish_run(self) -> None:
        self._running = False
        self.generate_button.configure(state="normal", text=GENERATE_LABEL)

    # ----------------------------------------------------------------- close --

    def _on_close(self) -> None:
        if self._running and not messagebox.askyesno(
            "Generation running",
            "A generation is still running. Exit anyway?",
            parent=self,
        ):
            return
        self.destroy()


def main(argv: Optional[list[str]] = None) -> int:
    arg_parser = argparse.ArgumentParser(
        description="CodeAgent_DeepSeek desktop dashboard"
    )
    arg_parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="open the window and close it automatically after a few seconds",
    )
    args = arg_parser.parse_args(argv)

    app = CodeAgentApp()
    if args.smoke_test:
        app.after(3500, app.destroy)
    app.mainloop()
    if args.smoke_test:
        print("SMOKE TEST OK: window opened and closed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
