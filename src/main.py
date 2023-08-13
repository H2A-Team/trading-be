from socket.socket_io import init_socket_app

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import router


def init_app() -> FastAPI:
    app = FastAPI()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        # allow_origins=settings.CORS_ALLOWED_ORIGINS,
        # allow_credentials=settings.CORS_IS_ALLOWED_CREDENTIALS,
        # allow_methods=settings.CORS_ALLOWED_METHODS,
        # allow_headers=settings.CORS_ALLOWED_HEADERS,
    )

    # REST API Routers
    app.include_router(router=router)

    # Socket
    # url prefix: "/ws"
    socketio_app = init_socket_app()
    # app.mount(settings.SOCKET_PREFIX, app=socketio_app)
    app.mount("/ws", app=socketio_app)

    return app


# FastAPI app includes both rest api and socker servers
backend_app = init_app()

if __name__ == "__main__":
    uvicorn.run(
        app="main:backend_app",
        # host=settings.SERVER_HOST,
        # port=settings.SERVER_PORT,
        # reload=settings.DEBUG,
        # log_level=settings.LOGGING_LEVEL,
    )
