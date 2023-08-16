import copy
from logging import basicConfig

from uvicorn.config import LOGGING_CONFIG

import settings

DEFAULT_UVICORN_LOGGING_CONFIG = copy.deepcopy(LOGGING_CONFIG)


def config_logging():
    # fastapi
    basicConfig(
        level=settings.LOGGING_LEVEL.upper(),
        filename=settings.LOGGING_FILENAME,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt=settings.LOGGING_DATETIME_FORMAT,
    )


def get_uvicorn_logging_config():
    DEFAULT_UVICORN_LOGGING_CONFIG["formatters"]["default"]["fmt"] = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    DEFAULT_UVICORN_LOGGING_CONFIG["formatters"]["default"]["datefmt"] = settings.LOGGING_DATETIME_FORMAT
    DEFAULT_UVICORN_LOGGING_CONFIG["formatters"]["access"]["fmt"] = '[%(asctime)s] [%(levelname)s] [%(name)s] - "%(request_line)s" %(status_code)s %(client_addr)s'
    DEFAULT_UVICORN_LOGGING_CONFIG["formatters"]["access"]["datefmt"] = settings.LOGGING_DATETIME_FORMAT
    return DEFAULT_UVICORN_LOGGING_CONFIG
