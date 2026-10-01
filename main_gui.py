"""
main_gui.py - CustomTkinter dashboard for the CodeAgent_DeepSeek pipeline.

A modern dark-mode desktop UI that:

    1. collects the DeepSeek API key, the two architecture documents and an
       output directory,
    2. parses the documents with ``parser.py``,
    3. runs the DeepSeek generation (``agent_core.py``) in a background thread
       so the window never freezes,
    4. streams live status / terminal output into the log panel, and
    5. shows success / error popups when a run finishes.

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
ACCENT = "#2F6FED"
ACCENT_HOVER = "#2559C9"

GENERATE_LABEL = "Generate Project Code"
GENERATE_RUNNING_LABEL = "Generating..."


class CodeAgentApp(ctk.CTk):
    """Main window of the desktop dashboard."""

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1000x780")
        self.minsize(880, 680)

        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._stage_count = 0
        self._parse_done = False
        self._total_steps = 1 + len(agent_core.DEFAULT_STAGES)

        self._build_ui()
        self.after(80, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._append_log(
            '[gui] Ready. Select the two architecture documents and an output '
            'folder, then click "Generate Project Code".'
        )
        self._append_log(
            "[gui] A JSON copy of the parsed architecture is saved into the "
            "output folder on each run."
        )

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header --------------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 2))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text=APP_TITLE,
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text=(
                "Parse architecture docs and generate the Space Fractions "
                "project (Node.js / Express, Docker, SQL, OpenAPI, tests) "
                "with the DeepSeek API."
            ),
            font=ctk.CTkFont(size=13),
            text_color=("gray35", "gray65"),
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Settings ------------------------------------------------------------
        settings = ctk.CTkFrame(self)
        settings.grid(row=1, column=0, sticky="ew", padx=20, pady=(12, 6))
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings, text="DeepSeek API Key", anchor="w").grid(
            row=0, column=0, padx=(14, 8), pady=(14, 6), sticky="w"
        )
        self.api_key_entry = ctk.CTkEntry(
            settings, placeholder_text="sk-...", show="*"
        )
        self.api_key_entry.grid(row=0, column=1, padx=8, pady=(14, 6), sticky="ew")

        self.show_key_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            settings,
            text="Show",
            variable=self.show_key_var,
            command=self._toggle_key_visibility,
            width=80,
        ).grid(row=0, column=2, padx=(8, 14), pady=(14, 6))

        self.doc_entry = self._path_row(
            settings,
            1,
            "Architecture Documentation",
            str(BASE_DIR / arch_parser.DOC_FILE_NAME),
            self._pick_documentation,
        )
        self.view_entry = self._path_row(
            settings,
            2,
            "Architecture Views",
            str(BASE_DIR / arch_parser.VIEW_FILE_NAME),
            self._pick_view,
        )
        self.out_entry = self._path_row(
            settings,
            3,
            "Target Output Directory",
            str(BASE_DIR / "output"),
            self._pick_output_dir,
            bottom_pady=(14, 14),
        )

        # Actions ---------------------------------------------------------------
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=20, pady=(6, 6))
        actions.grid_columnconfigure(2, weight=1)

        self.generate_button = ctk.CTkButton(
            actions,
            text=GENERATE_LABEL,
            height=46,
            width=250,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#FFFFFF",
            command=self._on_generate,
        )
        self.generate_button.grid(row=0, column=0, padx=(0, 12), pady=4)

        self.clear_button = ctk.CTkButton(
            actions,
            text="Clear Log",
            width=100,
            height=46,
            fg_color="transparent",
            border_width=1,
            command=self._clear_log,
        )
        self.clear_button.grid(row=0, column=1, padx=(0, 12), pady=4)

        self.progress = ctk.CTkProgressBar(actions, height=14)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(12, 0), pady=4)
        self.progress.set(0.0)

        # Log panel -------------------------------------------------------------
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(6, 4))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame,
            text="Execution Log",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))

        self.log_box = ctk.CTkTextbox(
            log_frame,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 12))
        self.log_box.configure(state="disabled")

        # Status bar ------------------------------------------------------------
        self.status_label = ctk.CTkLabel(
            self,
            text="Idle. Configure the inputs and click Generate Project Code.",
            anchor="w",
            text_color=("gray35", "gray60"),
        )
        self.status_label.grid(row=4, column=0, sticky="w", padx=22, pady=(0, 12))

    def _path_row(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label_text: str,
        initial: str,
        browse_command,
        bottom_pady=(6, 6),
    ) -> ctk.CTkEntry:
        ctk.CTkLabel(parent, text=label_text, anchor="w").grid(
            row=row, column=0, padx=(14, 8), pady=bottom_pady, sticky="w"
        )
        entry = ctk.CTkEntry(parent)
        entry.insert(0, initial)
        entry.grid(row=row, column=1, padx=8, pady=bottom_pady, sticky="ew")
        ctk.CTkButton(
            parent, text="Browse...", width=92, command=browse_command
        ).grid(row=row, column=2, padx=(8, 14), pady=bottom_pady)
        return entry

    # -------------------------------------------------------------- actions --

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
            self._set_entry(self.view_entry, path)

    def _pick_output_dir(self) -> None:
        initial = self.out_entry.get().strip() or str(BASE_DIR)
        path = filedialog.askdirectory(
            parent=self, title="Select target output directory", initialdir=initial
        )
        if path:
            self._set_entry(self.out_entry, path)

    @staticmethod
    def _set_entry(entry: ctk.CTkEntry, value: str) -> None:
        entry.delete(0, "end")
        entry.insert(0, value)

    # --------------------------------------------------------------- logging --

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        try:
            self.log_box.see("end")
        except AttributeError:  # pragma: no cover - fallback for older customtkinter
            self.log_box._textbox.see("end")  # noqa: SLF001
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------ generation --

    def _on_generate(self) -> None:
        if self._running:
            return

        api_key = self.api_key_entry.get().strip()
        doc_path = Path(self.doc_entry.get().strip())
        view_path = Path(self.view_entry.get().strip())
        out_value = self.out_entry.get().strip()

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
        self._current_out_dir = out_dir
        self._total_steps = 1 + len(agent_core.DEFAULT_STAGES)
        self.progress.set(0.0)
        self.generate_button.configure(state="disabled", text=GENERATE_RUNNING_LABEL)
        self.status_label.configure(text="Starting...")

        stamp = datetime.now().strftime("%H:%M:%S")
        self._append_log("-" * 64)
        self._append_log(f"[gui] Run started at {stamp}")
        self._append_log(f"[gui] Documentation : {doc_path}")
        self._append_log(f"[gui] Architecture  : {view_path}")
        self._append_log(f"[gui] Output folder : {out_dir}")

        self._worker = threading.Thread(
            target=self._worker_run,
            args=(doc_path, view_path, out_dir, api_key),
            daemon=True,
        )
        self._worker.start()

    def _worker_run(
        self, doc_path: Path, view_path: Path, out_dir: Path, api_key: str
    ) -> None:
        """Background worker: parse documents, then run DeepSeek generation."""
        q = self._queue
        try:
            q.put(("status", "Parsing architecture documents..."))
            q.put(("log", "[gui] Parsing architecture documents..."))
            payload = arch_parser.build_payload(doc_path, view_path)
            payload_path = arch_parser.save_payload(
                payload, out_dir / "architecture_payload.json"
            )
            q.put(("log", f"[gui] Payload saved to: {payload_path}"))

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
            
            res = agent_core.generate_project(
                payload,
                out_dir,
                api_key=api_key,
                log_callback=lambda msg: q.put(("log", msg))
            )

            q.put(("done", res))
        except Exception as e:
            q.put(("log", traceback.format_exc()))
            q.put(("error", f"{type(e).__name__}: {e}"))
    # ---------------------------------------------------------- queue drain --

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    self._track_progress(str(payload))
                elif kind == "progress":
                    self.progress.set(float(payload))
                elif kind == "status":
                    self.status_label.configure(text=str(payload))
                elif kind == "parsed":
                    self._parse_done = True
                    self._update_stage_progress()
                elif kind == "done":
                    self.progress.set(1.0)
                    self._finish_run()
                    if isinstance(payload, dict):
                        files = payload.get("files_written", [])
                        out_dir_str = payload.get("output_dir") or str(
                            getattr(self, "_current_out_dir", "")
                        )
                        used_fallback = bool(payload.get("used_fallback"))
                    else:
                        files = [str(p) for p in (payload or [])]
                        out_dir_str = str(getattr(self, "_current_out_dir", ""))
                        used_fallback = False
                    if used_fallback:
                        self.status_label.configure(text="Completed with fallback files.")
                        messagebox.showwarning(
                            "Generation finished with fallback files",
                            f"DeepSeek API was unavailable, so {len(files)} fallback "
                            f"sample file(s) were written into:\n{out_dir_str}\n\n"
                            "Check the log for the exact API error and retry with a "
                            "valid key / network connection.",
                            parent=self,
                        )
                    else:
                        self.status_label.configure(text="Completed successfully.")
                        messagebox.showinfo(
                            "Generation complete",
                            f"Generated {len(files)} file(s) into:\n{out_dir_str}",
                            parent=self,
                        )
                elif kind == "error":
                    self._finish_run()
                    self.status_label.configure(text="Failed.")
                    messagebox.showerror(
                        "Generation failed", str(payload), parent=self
                    )
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

    def _track_progress(self, line: str) -> None:
        if ": requesting" in line:
            match = re.search(r"stage '([^']+)'", line)
            if match:
                self.status_label.configure(
                    text=f"DeepSeek is generating: stage '{match.group(1)}'..."
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
