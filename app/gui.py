import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HF_HOME = _PROJECT_ROOT / "model" / "hf_home"
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_PROJECT_ROOT / "model" / "hf_cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(_HF_HOME / "modules"))

from app.services.search_engine import search


def get_config_dir():
    """PyInstaller 환경과 일반 환경 모두에서 config 디렉토리 경로 반환"""
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, "config")


# ─── 디자인 ─────────────────────────────────────────────────────────────────
PAD = 20
SIDEBAR_W = 240
GAP = 12

COLORS = {
    "bg": "#f1f5f9",
    "surface": "#ffffff",
    "surface_alt": "#f8fafc",
    "border": "#e2e8f0",
    "border_light": "#f1f5f9",
    "accent": "#0369a1",
    "accent_hover": "#0284c7",
    "accent_soft": "#e0f2fe",
    "text": "#0f172a",
    "text_secondary": "#475569",
    "text_muted": "#64748b",
    "link": "#0369a1",
    "link_hover": "#0284c7",
    "progress_trough": "#e2e8f0",
    "progress_bar": "#0369a1",
    "divider": "#e2e8f0",
}

FONTS = {
    "title": ("Segoe UI", 14, "bold"),
    "section": ("Segoe UI", 11, "bold"),
    "body": ("Segoe UI", 10),
    "body_bold": ("Segoe UI", 10, "bold"),
    "caption": ("Segoe UI", 9),
    "caption_bold": ("Segoe UI", 9, "bold"),
}


def _setup_styles(root: tk.Tk):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "App.Horizontal.TProgressbar",
        troughcolor=COLORS["progress_trough"],
        background=COLORS["progress_bar"],
        darkcolor=COLORS["progress_bar"],
        lightcolor=COLORS["progress_bar"],
        bordercolor=COLORS["border"],
        thickness=10,
    )


class SearchApp:
    def __init__(self, root, cfg: DictConfig):
        self.root = root
        self.cfg = cfg
        self.root.title("MIR 멀티모달 파일 검색기")
        self.root.geometry("1080x740")
        self.root.minsize(880, 600)
        self.root.configure(bg=COLORS["bg"])

        _setup_styles(self.root)

        # ─── 헤더 ─────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=COLORS["surface"], height=52)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        header_border = tk.Frame(
            self.root, bg=COLORS["border_light"], height=1
        )
        header_border.pack(fill=tk.X)

        header_inner = tk.Frame(
            header, bg=COLORS["surface"], padx=PAD, pady=12
        )
        header_inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            header_inner,
            text="MIR 멀티모달 파일 검색기",
            font=FONTS["caption_bold"],
            fg=COLORS["text_muted"],
            bg=COLORS["surface"],
        ).pack(side=tk.LEFT)
        tk.Label(
            header_inner,
            text="이미지·텍스트·음성·문서를 자연어로 검색합니다",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["surface"],
        ).pack(side=tk.RIGHT)

        # ─── 메인 컨텐츠 ───────────────────────────────────────────────────
        main = tk.Frame(self.root, bg=COLORS["bg"], padx=PAD, pady=PAD)
        main.pack(fill=tk.BOTH, expand=True)

        # 상단: 경로 + 검색
        top = tk.Frame(main, bg=COLORS["bg"])
        top.pack(fill=tk.X, pady=(0, GAP))

        path_row = tk.Frame(top, bg=COLORS["bg"])
        path_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            path_row,
            text="검색 경로",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg"],
        ).pack(anchor="w")
        path_box = tk.Frame(
            path_row,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        path_box.pack(fill=tk.X, pady=(4, 0))
        self.file_path_entry = tk.Entry(
            path_box,
            font=FONTS["body"],
            relief=tk.FLAT,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
        )
        self.file_path_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=12, pady=10
        )
        self.browse_button = tk.Button(
            path_box,
            text="파일 경로",
            font=FONTS["body"],
            command=self.browse_file,
            bg=COLORS["accent"],
            fg="white",
            activebackground=COLORS["accent_hover"],
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=18,
            pady=8,
        )
        self.browse_button.pack(side=tk.RIGHT, padx=6, pady=6)

        search_row = tk.Frame(top, bg=COLORS["bg"])
        search_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            search_row,
            text="검색어",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg"],
        ).pack(anchor="w")
        search_inner = tk.Frame(search_row, bg=COLORS["bg"])
        search_inner.pack(fill=tk.X, pady=(4, 0))
        query_box = tk.Frame(
            search_inner,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        query_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry = tk.Entry(
            query_box,
            font=FONTS["body"],
            relief=tk.FLAT,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
        )
        self.search_entry.pack(fill=tk.X, padx=12, pady=10)
        topk_frame = tk.Frame(search_inner, bg=COLORS["bg"])
        topk_frame.pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(
            topk_frame,
            text="Top K",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg"],
        ).pack(side=tk.LEFT)
        self.top_k_var = tk.StringVar(value="10")
        self.top_k_entry = tk.Spinbox(
            topk_frame,
            from_=1,
            to=100,
            textvariable=self.top_k_var,
            font=FONTS["body"],
            width=5,
            relief=tk.FLAT,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            buttonbackground=COLORS["surface"],
        )
        self.top_k_entry.pack(side=tk.LEFT, padx=(8, 0))
        self.search_button = tk.Button(
            search_inner,
            text="MIR",
            font=FONTS["body_bold"],
            command=self.search,
            bg=COLORS["accent"],
            fg="white",
            activebackground=COLORS["accent_hover"],
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=28,
            pady=10,
        )
        self.search_button.pack(side=tk.LEFT, padx=(18, 0))

        # ─── 중앙: 사이드바 + 결과 ─────────────────────────────────────────
        content = tk.Frame(main, bg=COLORS["bg"])
        content.pack(fill=tk.BOTH, expand=True, pady=(GAP, 0))

        sidebar = tk.Frame(
            content,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            width=SIDEBAR_W,
        )
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, PAD))
        sidebar.pack_propagate(False)

        sidebar_inner = tk.Frame(
            sidebar, bg=COLORS["surface"], padx=14, pady=14
        )
        sidebar_inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            sidebar_inner,
            text="파일 형식",
            font=FONTS["section"],
            fg=COLORS["text"],
            bg=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 12))

        self.ext_vars = {}
        self.type_vars = {}
        self.type_ext_map = {}
        allowed_map = (
            OmegaConf.to_container(self.cfg.allowed_extensions, resolve=True)
            or {}
        )
        for type_name, ext_list in allowed_map.items():
            t = str(type_name)
            self.type_ext_map[t] = ext_list
            self.type_vars[t] = tk.BooleanVar()
            tk.Checkbutton(
                sidebar_inner,
                text=t,
                variable=self.type_vars[t],
                font=FONTS["body_bold"],
                bg=COLORS["surface"],
                fg=COLORS["text"],
                activebackground=COLORS["surface"],
                activeforeground=COLORS["text"],
                selectcolor=COLORS["accent_soft"],
                anchor="w",
                command=lambda t=t: self._toggle_type(t),
            ).pack(anchor="w", pady=(6, 2))
            for ext in ext_list:
                var = tk.BooleanVar()
                self.ext_vars[ext] = (var, t)
                tk.Checkbutton(
                    sidebar_inner,
                    text=f"  {ext}",
                    variable=var,
                    font=FONTS["body"],
                    bg=COLORS["surface"],
                    fg=COLORS["text_secondary"],
                    activebackground=COLORS["surface"],
                    activeforeground=COLORS["text"],
                    selectcolor=COLORS["accent_soft"],
                    anchor="w",
                    command=lambda t=t: self._update_type_checkbox(t),
                ).pack(anchor="w", pady=2)
            tk.Frame(sidebar_inner, height=10, bg=COLORS["surface"]).pack()

        right = tk.Frame(content, bg=COLORS["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        progress_frame = tk.Frame(right, bg=COLORS["bg"])
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            progress_frame,
            textvariable=self.progress_var,
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg"],
            anchor="w",
        )
        self.progress_label.pack(fill=tk.X)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            style="App.Horizontal.TProgressbar",
            mode="determinate",
            maximum=100,
        )
        self.progress_bar.pack(fill=tk.X, pady=(6, 0))

        result_wrap = tk.Frame(
            right,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        result_wrap.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(
            result_wrap,
            bg=COLORS["surface"],
            troughcolor=COLORS["progress_trough"],
        )
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(
            result_wrap,
            wrap=tk.WORD,
            font=FONTS["body"],
            relief=tk.FLAT,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["accent_soft"],
            selectforeground=COLORS["text"],
            padx=14,
            pady=14,
            cursor="arrow",
            yscrollcommand=scroll.set,
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.result_text.yview)

        self.result_text.tag_configure(
            "header", font=FONTS["body_bold"], foreground=COLORS["text"]
        )
        self.result_text.tag_configure("text", foreground=COLORS["text"])
        self.result_text.tag_configure(
            "muted", foreground=COLORS["text_muted"]
        )
        self.result_text.tag_configure(
            "link", foreground=COLORS["link"], underline=True
        )
        self.result_text.tag_bind(
            "link",
            "<Enter>",
            lambda e: self.result_text.config(cursor="hand2"),
        )
        self.result_text.tag_bind(
            "link",
            "<Leave>",
            lambda e: self.result_text.config(cursor="arrow"),
        )

    def browse_file(self):
        path = filedialog.askdirectory()
        if path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, os.path.normpath(path))

    def _toggle_type(self, type_name: str):
        v = self.type_vars[type_name].get()
        for ext, (var, t) in self.ext_vars.items():
            if t == type_name:
                var.set(v)

    def _update_type_checkbox(self, type_name: str):
        ext_list = self.type_ext_map.get(type_name, [])
        n = sum(
            1
            for ext, (var, t) in self.ext_vars.items()
            if t == type_name and var.get()
        )
        self.type_vars[type_name].set(n == len(ext_list))

    def search(self):
        file_path = self.file_path_entry.get().strip()
        search_query = self.search_entry.get().strip()
        self.result_text.delete(1.0, tk.END)

        if not file_path:
            messagebox.showwarning("경고", "검색할 폴더 경로를 선택해주세요.")
            return
        if not search_query:
            messagebox.showwarning("경고", "검색어를 입력해주세요.")
            return
        extensions = [
            ext for ext, (var, _) in self.ext_vars.items() if var.get()
        ]
        if not extensions:
            messagebox.showwarning(
                "경고", "최소 하나의 파일 형식을 선택해주세요."
            )
            return
        allowed_map = (
            OmegaConf.to_container(self.cfg.allowed_extensions, resolve=True)
            or {}
        )
        target_extensions_map = {}
        sel = {str(e).lower() for e in extensions}
        if isinstance(allowed_map, dict):
            for type_, exts in allowed_map.items():
                exts_list = [str(e).lower() for e in (exts or [])]
                picked = [e for e in exts_list if e in sel]
                if picked:
                    target_extensions_map[str(type_)] = picked
        if not target_extensions_map:
            messagebox.showwarning("경고", "선택한 확장자가 설정에 없습니다.")
            return
        try:
            top_k = int(self.top_k_var.get())
            if top_k < 1:
                messagebox.showwarning("경고", "Top K는 1 이상이어야 합니다.")
                return
        except ValueError:
            messagebox.showwarning("경고", "Top K는 숫자로 입력해주세요.")
            return

        self.result_text.insert(tk.END, "검색 중...\n", "muted")
        self.search_button.config(state="disabled")
        self.root.update()

        def progress_callback(progress: int, message: str):
            self.root.after(
                0, lambda: self._update_progress(progress, message)
            )

        thread = threading.Thread(
            target=self._run_search,
            args=(
                search_query,
                file_path,
                target_extensions_map,
                extensions,
                top_k,
                progress_callback,
            ),
        )
        thread.start()

    def _run_search(
        self,
        query,
        target_dir,
        target_extensions_map,
        selected_extensions,
        top_k,
        progress_callback,
    ):
        try:
            results = search(
                query,
                target_dir,
                target_extensions_map,
                self.cfg,
                top_k=top_k,
                progress_callback=progress_callback,
            )
            self.root.after(
                0,
                lambda: self._display_results(
                    results, target_dir, query, selected_extensions
                ),
            )
        except Exception as e:
            self.root.after(0, lambda e=e: self._display_error(str(e)))

    def _update_progress(self, progress: int, message: str):
        self.progress_var.set(f"{message} ({progress}%)")
        self.progress_bar["value"] = progress
        self.root.update()

    def _display_results(self, results, target_dir, query, target_extensions):
        self.result_text.delete(1.0, tk.END)
        self.search_button.config(state="normal")
        self.progress_bar["value"] = 100
        self.progress_var.set("검색 완료 (100%)")
        """
        # 검색 정보를 한 줄로 표시 (가독성 향상)
        self.result_text.insert(tk.END, "검색 경로: ", "header")
        self.result_text.insert(tk.END, f"{target_dir}", "text")
        self.result_text.insert(tk.END, "  |  검색어: ", "header")
        self.result_text.insert(tk.END, f"{query}", "text")
        self.result_text.insert(tk.END, "  |  파일 형식: ", "header")
        self.result_text.insert(tk.END, f"{' '.join(target_extensions)}\n\n", "muted")
        """
        if not results:
            self.result_text.insert(tk.END, "검색 결과가 없습니다.\n", "muted")
            return
        if not isinstance(results, dict):
            self.result_text.insert(
                tk.END, "검색 결과 형식이 예상과 다릅니다.\n", "muted"
            )
            self.result_text.insert(tk.END, str(results))
            return

        type_titles = {
            "text": "텍스트",
            "image": "이미지",
            "voice": "음성",
            "docs": "문서",
        }
        link_index = 0
        for type_, items in results.items():
            if not items:
                continue
            title = type_titles.get(str(type_), str(type_))
            self.result_text.insert(tk.END, f"\n—— {title} ——\n\n", "header")
            for i, (file_path, score) in enumerate(items, 1):
                self.result_text.insert(tk.END, f"{i}. ", "muted")
                self._insert_file_link(file_path, link_index)
                self.result_text.insert(tk.END, "\n", "muted")
                link_index += 1

    def _insert_file_link(self, file_path, index):
        tag = f"link_{index}"
        self.result_text.tag_configure(
            tag, foreground=COLORS["link"], underline=True
        )
        self.result_text.insert(tk.END, file_path, tag)
        self.result_text.tag_bind(
            tag, "<Button-1>", lambda e, p=file_path: self._open_file(p)
        )
        self.result_text.tag_bind(
            tag, "<Enter>", lambda e: self.result_text.config(cursor="hand2")
        )
        self.result_text.tag_bind(
            tag, "<Leave>", lambda e: self.result_text.config(cursor="arrow")
        )

    def _open_file(self, file_path):
        try:
            os.startfile(file_path)
        except Exception as e:
            messagebox.showerror("오류", f"파일을 열 수 없습니다:\n{e}")

    def _display_error(self, error_msg):
        self.result_text.delete(1.0, tk.END)
        self.search_button.config(state="normal")
        self.progress_bar["value"] = 0
        self.progress_var.set("오류 발생")
        self.result_text.insert(
            tk.END, "검색 중 오류가 발생했습니다.\n\n", "header"
        )
        self.result_text.insert(tk.END, error_msg, "muted")
        messagebox.showerror(
            "오류", f"검색 중 오류가 발생했습니다:\n{error_msg}"
        )


def main() -> None:
    load_dotenv()
    config_dir = get_config_dir()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")
        cfg = cfg.default
        root = tk.Tk()
        SearchApp(root, cfg)
        root.mainloop()


if __name__ == "__main__":
    main()
