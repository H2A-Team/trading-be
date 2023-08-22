from fastapi import APIRouter

from settings import settings

from .v1.router import router as v1_router

router = APIRouter(prefix=settings.REST_API_PREFIX)

# api versioning
router.include_router(router=v1_router)
