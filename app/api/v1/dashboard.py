from fastapi import APIRouter

router = APIRouter()


@router.get("/stats")
async def get_stats():
    # TODO: 통화 통계 집계 조회
    raise NotImplementedError
