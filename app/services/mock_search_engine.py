from typing import List
import os


def search(query: str, root_path: str, extensions: List[str]):
    results = [
        os.path.join(root_path, "file1.txt"),
        os.path.join(root_path, "image1.jpg"),
    ]  # Mocked results
    return results
