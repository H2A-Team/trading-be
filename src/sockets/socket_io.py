from logging import getLogger

import socketio

from settings import settings

logger = getLogger('socketio_event_handlers')

sio_server = socketio.AsyncServer(async_mode='asgi', logger=True, cors_allowed_origins=settings.CORS_ALLOWED_ORIGINS)


@sio_server.event
async def connect(sid, environ):
    logger.info(f"{sid}: connected")


@sio_server.event
async def disconnect(sid):
    logger.info(f"{sid}, disconnected")
