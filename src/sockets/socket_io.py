from socketio import AsyncServer, ASGIApp

def init_socket_app() -> ASGIApp:
    socketio_server = AsyncServer(async_mode="asgi", cors_allowed_origins=[])
    socketio_app = ASGIApp(socketio_server=socketio_server)
    return socketio_app
