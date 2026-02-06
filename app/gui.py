import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys

import hydra
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from app.services.search_engine import search


def get_config_dir():
    """PyInstaller 환경과 일반 환경 모두에서 config 디렉토리 경로 반환"""
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, "config")


class SearchApp:
    def __init__(self, root, cfg: DictConfig):
        self.root = root
        self.cfg = cfg
        self.root.title("파일 검색")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        # 파일 경로 입력 영역
        self.file_path_entry = tk.Entry(root)
        self.file_path_entry.place(x=10, y=10, width=650, height=28)

        self.browse_button = tk.Button(
            root, text="파일찾기", command=self.browse_file
        )
        self.browse_button.place(x=670, y=8, width=100, height=32)

        # 검색어 입력 영역
        self.search_entry = tk.Entry(root)
        self.search_entry.place(x=10, y=50, width=450, height=28)

        self.search_button = tk.Button(root, text="검색", command=self.search)
        self.search_button.place(x=670, y=48, width=100, height=32)

        # Top K 입력 영역
        tk.Label(root, text="Top K:", bg="#f0f0f0").place(x=470, y=52)
        self.top_k_var = tk.StringVar(value="10")
        self.top_k_entry = tk.Spinbox(
            root, from_=1, to=100, textvariable=self.top_k_var, width=10
        )
        self.top_k_entry.place(x=520, y=50, width=80, height=28)

        # 타입별로 그룹핑된 체크박스 프레임 생성 (좌측)
        self.ext_vars = {}
        self.type_vars = {}  # 타입별 체크박스 변수
        self.type_ext_map = {}  # 타입별 확장자 매핑

        self.checkbox_outer_frame = tk.Frame(
            root, bg="white", relief="solid", bd=1
        )
        self.checkbox_outer_frame.place(x=10, y=95, width=200, height=600)

        allowed_map = (
            OmegaConf.to_container(self.cfg.allowed_extensions, resolve=True)
            or {}
        )
        y_offset = 0
        for type_name, ext_list in allowed_map.items():
            type_name_str = str(type_name)
            self.type_ext_map[type_name_str] = ext_list

            # 타입별 체크박스
            type_var = tk.BooleanVar()
            self.type_vars[type_name_str] = type_var
            type_checkbox = tk.Checkbutton(
                self.checkbox_outer_frame,
                text=type_name_str,
                variable=type_var,
                bg="white",
                anchor="w",
                font=("Arial", 10, "bold"),
                command=lambda t=type_name_str: self._toggle_type(t),
            )
            type_checkbox.place(x=5, y=y_offset)
            y_offset += 22

            # 확장자별 체크박스
            for ext in ext_list:
                var = tk.BooleanVar()
                self.ext_vars[ext] = (var, type_name_str)
                cb = tk.Checkbutton(
                    self.checkbox_outer_frame,
                    text=ext,
                    variable=var,
                    bg="white",
                    anchor="w",
                    command=lambda t=type_name_str: self._update_type_checkbox(
                        t
                    ),
                )
                cb.place(x=25, y=y_offset)
                y_offset += 20
            y_offset += 5  # 타입별 간격

        # 결과 표시 영역
        self.result_text = tk.Text(
            root, wrap=tk.WORD, relief="solid", bd=1, cursor="arrow"
        )
        self.result_text.place(x=220, y=95, width=770, height=600)

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

    def _toggle_type(self, type_name: str):
        """타입의 (전체) 체크박스를 클릭했을 때 해당 타입의 모든 확장자 토글"""
        is_checked = self.type_vars[type_name].get()
        for ext, (var, t) in self.ext_vars.items():
            if t == type_name:
                var.set(is_checked)

    def _update_type_checkbox(self, type_name: str):
        """확장자 체크박스가 변경되었을 때 타입의 (전체) 체크박스 상태 업데이트"""
        ext_list = self.type_ext_map.get(type_name, [])
        checked_count = 0
        for ext, (var, t) in self.ext_vars.items():
            if t == type_name and var.get():
                checked_count += 1

        # 모두 체크되면 (전체) 활성화, 모두 미체크되면 비활성화
        if checked_count == len(ext_list):
            self.type_vars[type_name].set(True)
        else:
            self.type_vars[type_name].set(False)

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
        extensions = [
            ext for ext, (var, _) in self.ext_vars.items() if var.get()
        ]
        if not extensions:
            messagebox.showwarning(
                "경고", "최소 하나의 파일 형식을 선택해주세요."
            )
            return

        # search_engine이 기대하는 타입별 확장자 맵으로 변환
        allowed_map = (
            OmegaConf.to_container(self.cfg.allowed_extensions, resolve=True)
            or {}
        )
        target_extensions_map = {}
        selected_set = {str(e).lower() for e in extensions}
        if isinstance(allowed_map, dict):
            for type_, exts in allowed_map.items():
                exts_list = [str(e).lower() for e in (exts or [])]
                picked = [e for e in exts_list if e in selected_set]
                if picked:
                    target_extensions_map[str(type_)] = picked

        if not target_extensions_map:
            messagebox.showwarning(
                "경고", "선택한 확장자가 config.allowed_extensions에 없습니다."
            )
            return

        # Top K 값 가져오기
        try:
            top_k = int(self.top_k_var.get())
            if top_k < 1:
                messagebox.showwarning("경고", "Top K는 1 이상이어야 합니다.")
                return
        except ValueError:
            messagebox.showwarning("경고", "Top K는 숫자여야 합니다.")
            return

        # 검색 중 표시
        self.result_text.insert(tk.END, "검색 중...\n")
        self.search_button.config(state="disabled")
        self.root.update()

        # 백그라운드에서 검색 실행
        thread = threading.Thread(
            target=self._run_search,
            args=(
                search_query,
                file_path,
                target_extensions_map,
                extensions,
                top_k,
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
    ):
        """백그라운드에서 검색 실행"""
        try:
            results = search(
                query, target_dir, target_extensions_map, self.cfg, top_k=top_k
            )
            # UI 업데이트는 메인 스레드에서 실행
            self.root.after(
                0,
                lambda: self._display_results(
                    results, target_dir, query, selected_extensions
                ),
            )
        except Exception as e:
            self.root.after(0, lambda e=e: self._display_error(str(e)))

    def _display_results(self, results, target_dir, query, target_extensions):
        """검색 결과 표시"""
        self.result_text.delete(1.0, tk.END)
        self.search_button.config(state="normal")

        header = f"검색 경로: {target_dir}\n"
        header += f"검색어: {query}\n"
        header += f"파일 형식: {' '.join(target_extensions)}\n"
        self.result_text.insert(tk.END, header)

        if not results:
            self.result_text.insert(tk.END, "\n검색 결과가 없습니다.\n")
            return

        # search_engine 반환: Dict[str, List[Tuple[path, score]]]
        if not isinstance(results, dict):
            self.result_text.insert(
                tk.END,
                "\n(경고) 검색 결과 형식이 예상과 다릅니다.\n",
            )
            self.result_text.insert(tk.END, str(results))
            return

        link_index = 0
        for type_, items in results.items():
            if not items:
                continue
            title = (
                "텍스트 파일"
                if str(type_) == "text"
                else "이미지 파일" if str(type_) == "image" else str(type_)
            )
            self.result_text.insert(
                tk.END, f"\n{'='*50}\n[{title}]\n{'='*50}\n\n"
            )

            for i, (file_path, score) in enumerate(items, 1):
                self.result_text.insert(tk.END, f"{i}. ")
                self._insert_file_link(file_path, link_index)
                self.result_text.insert(tk.END, f"  (score: {score:.4f})\n")
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


def main() -> None:
    config_dir = get_config_dir()
    load_dotenv(os.path.join(os.path.dirname(config_dir), ".env"))

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")
        cfg = cfg.default

        root = tk.Tk()
        app = SearchApp(root, cfg)
        root.mainloop()


if __name__ == "__main__":
    main()
