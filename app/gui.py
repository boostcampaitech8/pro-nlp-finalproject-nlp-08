import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from app.services.search_engine import search


class SearchApp:
    def __init__(self, root, cfg: DictConfig):
        self.root = root
        self.cfg = cfg
        self.root.title("파일 검색")
        self.root.geometry("850x530")
        self.root.configure(bg="#f0f0f0")

        # 파일 경로 입력 영역
        self.file_path_entry = tk.Entry(root)
        self.file_path_entry.place(x=20, y=70, width=620, height=30)

        self.browse_button = tk.Button(
            root, text="파일찾기", command=self.browse_file
        )
        self.browse_button.place(x=680, y=65, width=100, height=40)

        # 검색어 입력 영역
        self.search_entry = tk.Entry(root)
        self.search_entry.place(x=20, y=150, width=620, height=30)

        self.search_button = tk.Button(root, text="검색", command=self.search)
        self.search_button.place(x=680, y=145, width=100, height=40)

        # 체크박스 프레임 - config에서 동적으로 생성
        self.extension_list = []
        for exts in self.cfg.allowed_extensions:
            for ext in self.cfg.allowed_extensions[exts]:
                self.extension_list.append(ext)

        self.checkbox_frame = tk.Frame(root, bg="white", relief="solid", bd=1)
        checkbox_height = len(self.extension_list) * 25 + 10
        self.checkbox_frame.place(
            x=680, y=210, width=100, height=checkbox_height
        )

        # config의 allowed_extensions에서 체크박스 동적 생성
        self.ext_vars = {}
        for ext in self.extension_list:
            var = tk.BooleanVar()
            self.ext_vars[ext] = var
            checkbox = tk.Checkbutton(
                self.checkbox_frame,
                text=ext,
                variable=var,
                bg="white",
                anchor="w",
            )
            checkbox.pack(fill="x", padx=5)

        # 결과 표시 영역
        self.result_text = tk.Text(
            root, wrap=tk.WORD, relief="solid", bd=1, cursor="arrow"
        )
        self.result_text.place(x=20, y=300, width=760, height=200)

        # 링크 스타일 태그 설정
        self.result_text.tag_configure(
            "link", foreground="blue", underline=True
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
        """파일 탐색기 열기"""
        file_path = filedialog.askdirectory()
        if file_path:
            file_path = os.path.normpath(file_path)
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def search(self):
        """검색 실행"""
        file_path = self.file_path_entry.get()
        search_query = self.search_entry.get()

        # 결과 영역 초기화
        self.result_text.delete(1.0, tk.END)

        # 입력 검증
        if not file_path:
            messagebox.showwarning("경고", "파일 경로를 선택해주세요.")
            return
        if not search_query:
            messagebox.showwarning("경고", "검색어를 입력해주세요.")
            return

        # 선택된 확장자 목록 생성
        extensions = [ext for ext, var in self.ext_vars.items() if var.get()]
        if not extensions:
            messagebox.showwarning(
                "경고", "최소 하나의 파일 형식을 선택해주세요."
            )
            return

        # 검색 중 표시
        self.result_text.insert(tk.END, "검색 중...\n")
        self.search_button.config(state="disabled")
        self.root.update()

        # 백그라운드에서 검색 실행
        thread = threading.Thread(
            target=self._run_search,
            args=(search_query, file_path, extensions),
        )
        thread.start()

    def _run_search(self, query, target_path, extensions):
        """백그라운드에서 검색 실행"""
        try:
            results = search(
                query=query,
                target_path=target_path,
                extensions=extensions,
                top_k=self.cfg.search.top_k,
                cache_path=self.cfg.cache_path,
                cfg=self.cfg,
            )
            # UI 업데이트는 메인 스레드에서 실행
            self.root.after(
                0,
                lambda: self._display_results(
                    results, target_path, query, extensions
                ),
            )
        except Exception as e:
            self.root.after(0, lambda: self._display_error(str(e)))

    def _display_results(self, results, target_path, query, extensions):
        """검색 결과 표시"""
        self.result_text.delete(1.0, tk.END)
        self.search_button.config(state="normal")

        header = f"검색 경로: {target_path}\n"
        header += f"검색어: {query}\n"
        header += f"파일 형식: {' '.join(extensions)}\n"
        self.result_text.insert(tk.END, header)

        if not results:
            self.result_text.insert(tk.END, "\n검색 결과가 없습니다.\n")
            return

        # 결과를 텍스트와 이미지로 분류
        text_results = [
            f
            for f in results
            if f.lower().endswith(
                tuple(OmegaConf.to_container(self.cfg.allowed_extensions.text))
            )
        ]
        image_results = [
            f
            for f in results
            if f.lower().endswith(
                tuple(
                    OmegaConf.to_container(self.cfg.allowed_extensions.image)
                )
            )
        ]

        link_index = 0

        # 텍스트 결과 표시
        if text_results:
            self.result_text.insert(
                tk.END, f"\n{'='*50}\n[텍스트 파일]\n{'='*50}\n\n"
            )
            for i, file_path in enumerate(text_results, 1):
                self.result_text.insert(tk.END, f"{i}. ")
                self._insert_file_link(file_path, link_index)
                self.result_text.insert(tk.END, "\n")
                link_index += 1

        # 이미지 결과 표시
        if image_results:
            self.result_text.insert(
                tk.END, f"\n{'='*50}\n[이미지 파일]\n{'='*50}\n\n"
            )
            for i, file_path in enumerate(image_results, 1):
                self.result_text.insert(tk.END, f"{i}. ")
                self._insert_file_link(file_path, link_index)
                self.result_text.insert(tk.END, "\n")
                link_index += 1

    def _insert_file_link(self, file_path, index):
        """파일 경로를 클릭 가능한 링크로 삽입"""
        link_tag = f"link_{index}"
        self.result_text.tag_configure(
            link_tag, foreground="blue", underline=True
        )
        self.result_text.insert(tk.END, file_path, link_tag)
        self.result_text.tag_bind(
            link_tag,
            "<Button-1>",
            lambda e, path=file_path: self._open_file(path),
        )
        self.result_text.tag_bind(
            link_tag,
            "<Enter>",
            lambda e: self.result_text.config(cursor="hand2"),
        )
        self.result_text.tag_bind(
            link_tag,
            "<Leave>",
            lambda e: self.result_text.config(cursor="arrow"),
        )

    def _open_file(self, file_path):
        """파일 열기"""
        try:
            os.startfile(file_path)
        except Exception as e:
            messagebox.showerror("오류", f"파일을 열 수 없습니다:\n{e}")

    def _display_error(self, error_msg):
        """에러 메시지 표시"""
        self.result_text.delete(1.0, tk.END)
        self.search_button.config(state="normal")
        self.result_text.insert(tk.END, f"검색 중 오류 발생:\n{error_msg}")
        messagebox.showerror(
            "오류", f"검색 중 오류가 발생했습니다:\n{error_msg}"
        )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    root = tk.Tk()
    cfg = cfg.default
    app = SearchApp(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
