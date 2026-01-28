from fastapi import APIRouter, Depends, Request, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict, Any
from app.services.mock_search_engine import search as search_service

router = APIRouter()


class SearchRequest(BaseModel):
    """
    검색 API에서 사용하는 요청 파라미터 스키마입니다.

    Attributes:
        query (str): 검색 질의 문자열
        root_path (str): 검색 대상 루트 경로
        extensions (Optional[List[str]]): 검색 대상 확장자 목록(예: [".pdf", "txt"])
    """

    query: str = Field(..., description="검색 질의 문자열")
    root_path: str = Field(..., description="검색 대상 루트 경로")
    extensions: Optional[List[str]] = Field(
        default=None,
        description="검색 대상 확장자 목록(점 포함/미포함 모두 허용). 미입력 시 서버 모든 허용 확장자 사용",
    )


def _normalize_ext(ext: str) -> str:
    """
    확장자 문자열을 표준 형태로 정규화합니다.

    Args:
        ext (str): 확장자 문자열(예: 'pdf', '.PDF', '  txt ')

    Returns:
        str: 소문자 및 점(.)을 포함한 확장자(예: '.pdf')
    """
    normalized = ext.strip().lower()
    if not normalized:
        return normalized
    return normalized if normalized.startswith(".") else f".{normalized}"


def _resolve_target_extensions(
    requested_extensions: Optional[List[str]],
    allowed_extensions: Set[str],
) -> List[str]:
    """
    요청 확장자와 서버 허용 확장자를 바탕으로 실제 검색 대상 확장자를 확정합니다.

    Args:
        requested_extensions (Optional[List[str]]): 사용자가 요청한 확장자 목록
        allowed_extensions (Set[str]): 서버에서 허용하는 확장자 집합

    Returns:
        List[str]: 실제 검색에 사용할 확장자 목록(정규화/중복 제거)

    Raises:
        HTTPException: 허용되지 않은 확장자가 포함된 경우(400)
    """
    if requested_extensions:
        target_exts = {_normalize_ext(ext) for ext in requested_extensions if ext is not None}
    else:
        target_exts = set(allowed_extensions)

    # 빈 문자열 확장자는 의미가 없으므로 제거
    target_exts.discard("")

    invalid_exts = target_exts - allowed_extensions
    if invalid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extensions: {', '.join(sorted(invalid_exts))}",
        )

    return sorted(target_exts)


def get_search_request(
    query: str = Query(...),
    root_path: str = Query(...),
    extensions: Optional[List[str]] = Query(None),
) -> SearchRequest:
    """
    쿼리 파라미터를 `SearchRequest`로 변환합니다.
    """
    return SearchRequest(query=query, root_path=root_path, extensions=extensions)


@router.get("/search")
def search(
    request: Request,
    params: SearchRequest = Depends(get_search_request),
) -> Dict[str, Any]:
    """
    검색 API 엔드포인트.

    Args:
        request (Request): FastAPI 요청 객체(앱 상태 접근 목적)
        params (SearchRequest): 검색 파라미터

    Returns:
        Dict[str, Any]: 검색 결과 응답
    """
    config = request.app.state.config
    allowed_extensions = {_normalize_ext(e) for e in getattr(config, "allowed_extensions", [])}
    allowed_extensions.discard("")

    target_exts = _resolve_target_extensions(params.extensions, allowed_extensions)

    search_args = params.dict()
    search_args["extensions"] = target_exts

    results = search_service(**search_args)
    return {"status": "success", "results": results}