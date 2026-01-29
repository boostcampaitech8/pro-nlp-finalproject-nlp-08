# 파일 스캔 및 절대 경로 관리
from pathlib import Path
from typing import List, Dict

class FileManager:
    def __init__(self, db, root_dir: str):
        '''
        FileManager 클래스 초기화
        - 로컬 파일 시스템을 스탬하고, pkl 파일에 저장된 파일 목록과 동기화 작업을 담당함.
        
        Args:
            db: 데이터베이스 객체
            root_dir (str): 로컬 파일시스템 스캔할 루트 디렉토리 경로
        '''
        self.db = db
        self.root_dir = root_dir
        
    def scan_directory(self, root_path: str) -> List[str]:
        '''
        주어진 루트 경로를 기준으로 모든 하위 디렉토리와 파일을 스캔함.
            
        Args:
            root_path (str): 스캔할 루트 디렉토리 경로
            
        Returns:
            List[str]: 모든 파일의 절대 경로 리스트
        '''
        file_paths = []
            
        for file in Path(root_path).rglob('*'):
            if file.is_file():
                file_paths.append(str(file))
            
        return file_paths
        
    def get_file_mtime(self, file_path: str) -> float:
        '''
        파일 경로에 대한 수정 시간을 반환함.
            
        Args:
            file_path (str): 파일 경로
            
        Returns:
            float: 파일이 저장된 시간 반환
        '''
        stat =  Path(file_path).stat()
        mtime = stat.st_mtime
        return mtime
        
    def sync_and_filter(self) -> List[str]:
        '''
        실제 로컬 파일 목록과 cache되어 있는 목록들을 비교하여 동기화 작업을 수행함.
        - 로컬에서 삭제되거나 수정된 파일은 캐시에서 제거
        - 로컬에서 새롭게 추가된 파일 혹은 수정된 파일은 임베딩이 필요한 목록에 추가
            
        Returns:
            List[str]: 새롭게 추가되거나 수정된 파일들의 절대 경로 리스트
        '''
            
        local_files = self.scan_directory(self.root_dir)
        db_files = self.db.get_keys_cache()
        need_embed_files = []
            
        # 로컬 파일과 DB 파일 비교    
        for file_path in local_files:
            local_mtime = self.get_file_mtime(file_path)
            
            # 새로운 파일
            if file_path not in db_files:
                need_embed_files.append(file_path)
            # 수정된 파일
            else:
                db_mtime = self.db.get_mtime_cache(file_path)
                if local_mtime != db_mtime:
                    print(f"File modified: {file_path} check mtime {local_mtime}  vs {type(db_mtime)} {db_mtime}")
                    need_embed_files.append(file_path)
                    self.db.delete_cache(file_path)
                    
        # DB에서 삭제된 파일 제거
        delete_cache_list = set(db_files) - set(local_files)
        if delete_cache_list:
            for db_file in delete_cache_list:
                self.db.delete_cache(db_file)
                
        return need_embed_files