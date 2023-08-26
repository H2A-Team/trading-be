# import asyncio
# import json
# import sys
# import threading
# import time
from logging import getLogger

import socketio
# from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager

from settings import settings
from utilities.binance import normalize_stream_name

# loggers
socket_logger = getLogger('socketio_event_handlers')
stream_logger = getLogger("binance_market_stream")

# key: stream name
# value: list of socket ids
stream_mappings: dict[str, list[str]] = {}
for stream_name in settings.BINANCE_MARKET_STREAM_NAMES:
    stream_mappings.setdefault(stream_name, [])

# socker servers
sio_server = socketio.AsyncServer(
    async_mode='asgi',
    logger=socket_logger,
    cors_allowed_origins=settings.SOCKET_CORS_ALLOWED_ORIGINS,
)

# create instance of BinanceWebSocketApiManager and provide the function for stream processing
# binance_websocket_api_manager = BinanceWebSocketApiManager(exchange=settings.BINANCE_SOCKET_EXCHANGE)


@sio_server.event
async def connect(sid, environ):
    socket_logger.info(f"{sid}: connected")


@sio_server.event
async def disconnect(sid):
    socket_logger.info(f"{sid}, disconnected")
    for stream_name in stream_mappings:
        if sid in stream_mappings[stream_name]:
            stream_mappings[stream_name].remove(sid)
    # socket_logger.info(f"stream_mappings:disconnect {stream_mappings}")


@sio_server.on("subscribe")
async def join_room(sid, room):
    room = normalize_stream_name(room)
    sio_server.enter_room(sid, room)

    if room in stream_mappings:
        stream_mappings[room].append(sid)
    # socket_logger.info(f"stream_mappings:subscribe {stream_mappings[room]}")


@sio_server.on("unsubscribe")
async def leave_room(sid, room):
    room = normalize_stream_name(room)
    sio_server.leave_room(sid, room)

    if room in stream_mappings and sid in stream_mappings[room]:
        stream_mappings[room].remove(sid)
    # socket_logger.info(f"stream_mappings:unsubscribe {stream_mappings[room]}")


async def broadcast_to_stream(event, room, data):
    await sio_server.emit(event=event, room=room, data=data)


# ----------------------------------------------------------------------------------------------------------------
# BINANCE WEBSOCKET STREAM LOGIC


# def market_stream_data_callback(binance_websocket_api_manager, _stream_logger, stream_mappings):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     while True:
#         if binance_websocket_api_manager.is_manager_stopping():
#             _stream_logger.info("stop stream thread")
#             loop.close()
#             sys.exit(0)

#         oldest_stream_data_from_stream_buffer = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
#         if oldest_stream_data_from_stream_buffer is False:
#             time.sleep(0.01)
#         else:
#             try:
#                 buffer_data = json.loads(str(oldest_stream_data_from_stream_buffer))
#                 stream_name = buffer_data.get("stream", "")
#                 _stream_logger.info(buffer_data)

#                 if stream_name in settings.BINANCE_MARKET_STREAM_NAMES and len(stream_mappings[stream_name]) > 0:
#                     stream_data = buffer_data.get("data")
#                     candle_data = stream_data.get("k")
#                     data_to_sent = {
#                         "binanceEventTimestamp": stream_data["E"],
#                         "symbol": stream_data["s"],
#                         "startIntervalTimestamp": candle_data["t"],
#                         "endIntervalTimestamp": candle_data["T"],
#                         "interval": candle_data["i"],
#                         "openPrice": candle_data["o"],
#                         "closePrice": candle_data["c"],
#                         "highPrice": candle_data["h"],
#                         "lowPrice": candle_data["l"],
#                         "volume": candle_data["v"],
#                     }

#                     loop.run_until_complete(broadcast_to_stream("realtime_candle", stream_name, data_to_sent))
#             except KeyError:
#                 # Any kind of error...
#                 # not able to process the data? write it back to the stream_buffer
#                 binance_websocket_api_manager.add_to_stream_buffer(oldest_stream_data_from_stream_buffer)


# # start a worker process to process to move the received stream_data from the stream_buffer to a print function
# worker_thread = threading.Thread(
#     target=market_stream_data_callback,
#     args=(
#         binance_websocket_api_manager,
#         stream_logger,
#         stream_mappings,
#     ),
# )
# worker_thread.start()

# stream_id = binance_websocket_api_manager.create_stream(
#     channels=settings.BINANCE_MARKET_CHANNELS,
#     markets=settings.BINANCE_MARKET_SYMBOLS,
# )

# # binance_websocket_api_manager.subscribe_to_stream(channels=['kline_1m'], markets=['BTCUSDT'])
# request_id = binance_websocket_api_manager.get_stream_subscriptions(stream_id)

# while binance_websocket_api_manager.get_result_by_request_id(request_id) is False:
#     stream_logger.info("Wait to receive the result!")
#     time.sleep(2)

# stream_logger.info("Request result: " + str(binance_websocket_api_manager.get_result_by_request_id(request_id)))

# start a restful api server to report the current status to 'tools/icinga/check_binance_websocket_manager' which can be
# used as a check_command for ICINGA/Nagios
# binance_websocket_api_manager.start_monitoring_api()
