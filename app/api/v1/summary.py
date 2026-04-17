from fastapi import APIRouter

router = APIRouter()


@router.get("/{call_id}")
async def get_summary(call_id: str):
    # TODO: call_summaries 테이블에서 조회
    raise NotImplementedError
