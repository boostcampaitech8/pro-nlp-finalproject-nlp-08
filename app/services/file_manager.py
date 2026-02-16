# 파일 스캔 및 절대 경로 관리
from typing import List, Dict
from collections import defaultdict
import os


class FileManager:
    def __init__(
        self, target_dir: str, target_extensions: Dict[str, List[str]]
    ):
        """
        FileManager 초기화

        Args:
            target_dir (str): 파일 스캔을 시작할 루트 디렉토리 경로
            target_extensions (Dict[str, List[str]]): 확장자별 파일 타입 매핑
                예: {'image': ['.png', '.jpg'], 'text': ['.txt', '.md']}
        """
        self.target_dir = target_dir
        self.extensions_type_map = {
            ext: type_
            for type_, exts in target_extensions.items()
            for ext in exts
        }

    def scan_directory(self) -> Dict[str, List[str]]:
        """
        주어진 루트 경로를 기준으로 모든 하위 디렉토리와 파일을 스캔함.

        Args:
            target_path (str): 스캔할 루트 디렉토리 경로

        Returns:
            Dict[str, List[str]]: 타입별(예: 'image', 'text') 파일 경로 리스트
        """
        file_paths = defaultdict(list)
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                file_type = self.extensions_type_map.get(ext)
                if file_type:
                    full_path = os.path.join(root, file)
                    file_paths[file_type].append(full_path)

        return dict(file_paths)
