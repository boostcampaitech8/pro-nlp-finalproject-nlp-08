import requests
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig

TEST_DIR = str(Path(__file__).parent.resolve() / "test_dir")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def test_search_api(cfg: DictConfig):
    config = cfg.default
    base_url = f"http://{config.server.host}:{config.server.port}"

    # Test case 1: Valid search request
    params = {
        "query": "cat",
        "root_path": TEST_DIR,
        "extensions": ["txt", "jpg"]
    }
    response = requests.get(f"{base_url}/search", params=params)
    assert response.status_code == 200
    data = response.json()
    print(data)
    assert data["status"] == "success"
    assert isinstance(data["results"], list)
    
    # Test case 2: Invalid extension
    params = {
        "query": "cat",
        "root_path": TEST_DIR,
        "extensions": ["exe"]
    }
    response = requests.get(f"{base_url}/search", params=params)
    assert response.status_code == 400
    data = response.json()
    print(data)
    assert "Invalid extensions" in data["detail"]
    
    # Test case 3: No extensions provided (should use all allowed)
    params = {
        "query": "cat",
        "root_path": TEST_DIR
    }
    response = requests.get(f"{base_url}/search", params=params)
    assert response.status_code == 200
    data = response.json()
    print(data)
    assert data["status"] == "success"
    assert isinstance(data["results"], list)

if __name__ == "__main__":
    test_search_api()
