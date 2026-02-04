import hydra
import uvicorn
from omegaconf import DictConfig
from fastapi import FastAPI

from app.api.routes import search


def create_app(cfg: DictConfig) -> FastAPI:
    """
    FastAPI 애플리케이션을 생성하고 라우터/설정을 주입합니다.

    Args:
        cfg (DictConfig): 앱 전역 설정(서버/허용 확장자 등)

    Returns:
        FastAPI: 구성 완료된 FastAPI 인스턴스
    """
    app = FastAPI()
    app.state.config = cfg
    app.include_router(search.router)
    return app


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra 설정을 로드하고 FastAPI 서버를 실행합니다.
    """
    app_cfg = cfg.default
    app = create_app(app_cfg)
    uvicorn.run(app, host=app_cfg.server.host, port=app_cfg.server.port)


if __name__ == "__main__":
    main()
