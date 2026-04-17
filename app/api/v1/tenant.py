from fastapi import APIRouter

router = APIRouter()


@router.get("/{tenant_id}")
async def get_tenant(tenant_id: str):
    # TODO: tenants 테이블에서 조회
    raise NotImplementedError


@router.post("/")
async def create_tenant():
    # TODO: tenants 테이블에 INSERT
    raise NotImplementedError
