from app.services.search_engine import search
import os
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Hugging Face 캐시를 프로젝트 내부로 고정합니다.
# (가중치 cache_dir와 별개로, trust_remote_code 모듈은 HF_MODULES_CACHE를 사용)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HF_HOME = _PROJECT_ROOT / "model" / "hf_home"
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_PROJECT_ROOT / "model" / "hf_cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(_HF_HOME / "modules"))



query = "Greeting!"
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
target_path = os.path.join(ROOT_DIR, "test", "test_dir")
target_extensions = {
    "text": [".txt", ".md"],
    "image": [".jpg", ".png"],
}
cfg = OmegaConf.load(os.path.join(ROOT_DIR, "config", "config.yaml")).default


try:
    results = search(
        query=query,
        target_dir=target_path,
        target_extensions=target_extensions,
        cfg=cfg,
    )
except Exception as e:
    print(f"Error during search: {e}")
    import traceback

    traceback.print_exc()
    raise

print("Search Results:")
for type_, items in results.items():
    print(f"Type: {type_}")
    for file_path, score in items:
        print(f"  File: {file_path}, Score: {score}")
