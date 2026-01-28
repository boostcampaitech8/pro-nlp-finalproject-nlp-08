# 캐시 조회 저장 삭제 업데이트
import pickle


class VectorStore:
    """
    pickle 기반 VectorStore 구현 및 조회, 삽입, 삭제 기능
    지정된 경로의 pickle 파일을 읽어 데이터베이스를 메모리에 로드

    Args:
        db_path (str): 캐시 데이터가 저장된 pickle 파일 경로
    
    Returns:
        None
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.data={}

        try:
            with open(self.db_path, 'rb') as f:
                self.data = pickle.load(f)
        except EOFError:
            pass

    def get_mtime_cache(self, key):
        """
        데이터베이스의 특정 key에 대한 mtime을 반환

        Args:
            key (str): mtime을 가져올 key값

        Returns:
            Float: 검색할 단일 데이터의 mtime
        """
        return self.data[key][0]

    def get_keys_cache(self):
        """
        데이터베이스의 모든 key를 반환

        Args:
            None

        Returns:
            List[str]: 모든 데이터의 key List
        """
        return self.data.keys()
    
    def get_values_cache(self):
        """
        데이터베이스의 모든 value를 반환

        Args:
            None

        Returns:
            List[List[Float, List[Float]]]: 모든 데이터의 value List
        """
        return self.data.values()
    
    def delete_cache(self, del_key):
        """
        데이터베이스의 특정 key에 해당하는 데이터 삭제

        Args:
            del_key (str): 특정 데이터에 해당하는 key값

        Returns:
            None
        """
        if del_key in self.data:
            del self.data[del_key]
    
    def create_cache(self, create_key, mtime, create_embed):
        """
        데이터베이스의 데이터 추가

        Args:
            create_key (key): 데이터의 key값(임베딩한 문서의 절대 경로)
            mtime (Float): 데이터 최종 수정 시간(수정된 데이터인지 확인 목적)
            create_embed (List[Float]): 삽입할 데이터 문서의 임베딩

        Returns:
            None
        """
        self.data[create_key] = [mtime, create_embed]
    
    def save_cache(self):
        """
        데이터베이스 변경 사항을 기존 저장 공간에 덮어씌워 저장

        Args:
            None

        Returns:
            None
        """
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)