from logging import getLogger

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import settings
from api.router import router
from logging_config import config_logging, get_uvicorn_logging_config
from sockets.socket_io import init_socket_app


def bootstrap_config():
    config_logging()


def init_app() -> FastAPI:
    app = FastAPI()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_methods=settings.CORS_ALLOWED_METHODS,
    )

    # REST API Routers
    app.include_router(router=router)

    # Socket
    socketio_app = init_socket_app()
    app.mount(settings.SOCKET_PREFIX, app=socketio_app)

    return app


bootstrap_config()
backend_app = init_app()

if __name__ == "__main__":
    uvicorn.run(
        app="main:backend_app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOGGING_LEVEL.lower(),
        log_config=get_uvicorn_logging_config(),
    )
