from socketio import ASGIApp, AsyncServer

import settings


def init_socket_app() -> ASGIApp:
    socketio_server = AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=settings.CORS_ALLOWED_ORIGINS,
    )
    return ASGIApp(socketio_server=socketio_server)
