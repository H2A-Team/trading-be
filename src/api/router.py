from fastapi import APIRouter
from v1 import v1_router

# router = APIRouter(prefix=settings.REST_API_PREFIX)
router = APIRouter(prefix="/api")

# api versioning
router.include_router(router=v1_router)
