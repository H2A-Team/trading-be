from logging import basicConfig

# import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import router
from settings import settings
# from sockets.socket_io import sio_server


def bootstrap_config():
    basicConfig(**settings.logging_config)


def init_app() -> FastAPI:
    app = FastAPI(**settings.app_config)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_methods=settings.CORS_ALLOWED_METHODS,
    )

    # REST API Routers
    app.include_router(router=router)

    # Socket
    # socketio_app = socketio.ASGIApp(sio_server, socketio_path=settings.SOCKET_SOCKETIO_PATH)
    # app.mount(settings.SOCKET_PREFIX, socketio_app)

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
        log_config=settings.uvicorn_logging_config,
    )
