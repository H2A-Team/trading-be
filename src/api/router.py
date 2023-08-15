from fastapi import APIRouter
from .v1.router import v1_router

import settings

router = APIRouter(prefix=settings.REST_API_PREFIX)

# api versioning
router.include_router(router=v1_router)
