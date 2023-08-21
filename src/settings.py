import copy
import pathlib

from decouple import AutoConfig, Csv
from uvicorn.config import LOGGING_CONFIG

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
get_config = AutoConfig(search_path=str(ROOT_DIR) + "/.env")

PREFIX_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]"
_PREDICTION_MODEL_LOCATION = str(pathlib.Path(__file__).resolve()) + "/predictions/trained-models"


class Settings:
    def __init__(self):
        self.__default_uvicorn_logging_config = copy.deepcopy(LOGGING_CONFIG)

    # fastapi app
    DEBUG = get_config("DEBUG", default=False, cast=bool)
    APP_TITLE = get_config("APP_TITLE", default="H2A Trading")
    APP_SUMMARY = get_config("APP_SUMMARY", default="")
    APP_DESCRIPTION = get_config("APP_DESCRIPTION", default="")
    APP_VERSION = get_config("APP_VERSION", default="0.0.1")

    # server
    SERVER_HOST = get_config("SERVER_HOST", default="localhost")
    SERVER_PORT = get_config("SERVER_PORT", default=8000, cast=int)

    # logging
    LOGGING_LEVEL = get_config("LOGGING_LEVEL", default="INFO")
    LOGGING_FILENAME = get_config("LOGGING_FILENAME", default=None)
    LOGGING_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"
    LOGGING_DEFAULT_LOG_FORMAT = f"{PREFIX_LOG_FORMAT} - %(message)s"
    LOGGING_ACCESS_LOG_FORMAT = f'{PREFIX_LOG_FORMAT} - "%(request_line)s" %(status_code)s %(client_addr)s'

    # cors
    CORS_ALLOWED_ORIGINS = get_config("CORS_ALLOWED_ORIGINS", default="*", cast=Csv())
    CORS_ALLOWED_METHODS = get_config("CORS_ALLOWED_METHODS", default="*", cast=Csv())

    # rest api
    REST_API_PREFIX = get_config("REST_API_PREFIX", default="/api")
    REST_API_OPENAPI_URL = get_config("REST_API_OPENAPI_URL", default="/openapi.json")
    REST_API_SWAGGER_URL = get_config("REST_API_SWAGGER_URL", default="/docs")
    REST_API_REDOC_URL = get_config("REST_API_REDOC_URL", default="/redoc")

    # socket
    SOCKET_PREFIX = get_config("SOCKET_PREFIX", default="/socket")
    SOCKET_SOCKETIO_PATH = get_config("SOCKET_SOCKETIO_PATH", default="socketio")

    # core machine learning / predictions
    PREDICTION_MODEL_LOCATION = _PREDICTION_MODEL_LOCATION
    PREDICTION_TIME_STEPS = 60

    @property
    def app_config(self):
        return {
            "title": self.APP_TITLE,
            "version": self.APP_VERSION,
            "debug": self.DEBUG,
            "description": self.APP_DESCRIPTION,
            "docs_url": self.REST_API_SWAGGER_URL,
            "openapi_url": self.REST_API_OPENAPI_URL,
            "redoc_url": self.REST_API_REDOC_URL,
            "api_prefix": self.REST_API_PREFIX,
        }

    @property
    def logging_config(self) -> dict[str, str]:
        config = {
            "level": self.LOGGING_LEVEL.upper(),
            "format": self.LOGGING_DEFAULT_LOG_FORMAT,
            "datefmt": self.LOGGING_DATETIME_FORMAT,
        }

        if self.LOGGING_FILENAME:
            config["filename"] = self.LOGGING_FILENAME

        return config

    @property
    def uvicorn_logging_config(self):
        self.__default_uvicorn_logging_config["formatters"]["default"]["fmt"] = settings.LOGGING_DEFAULT_LOG_FORMAT
        self.__default_uvicorn_logging_config["formatters"]["default"]["datefmt"] = settings.LOGGING_DATETIME_FORMAT
        self.__default_uvicorn_logging_config["formatters"]["access"]["fmt"] = settings.LOGGING_ACCESS_LOG_FORMAT
        self.__default_uvicorn_logging_config["formatters"]["access"]["datefmt"] = settings.LOGGING_DATETIME_FORMAT
        return self.__default_uvicorn_logging_config


# global settings variable
settings = Settings()
