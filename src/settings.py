from decouple import config, Csv

# general
DEBUG = config('DEBUG', default=False, cast=bool)

# server
SERVER_HOST = config('SERVER_HOST', default='localhost')
SERVER_PORT = config('SERVER_PORT', default=8000, cast=int)

# logging
LOGGING_LEVEL = config('LOGGING_LEVEL', default='INFO')
LOGGING_FILENAME = config('LOGGING_FILENAME', default=None)
LOGGING_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"

# cors
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default="*", cast=Csv())
CORS_ALLOWED_METHODS = config('CORS_ALLOWED_METHODS', default="*", cast=Csv())

# rest api
REST_API_PREFIX = config('REST_API_PREFIX', default='/api')
REST_API_SWAGGER_URL = config('REST_API_SWAGGER_URL', default='/docs')
REST_API_REDOC_URL = config('REST_API_REDOC_URL', default='/redoc')

# socket
SOCKET_PREFIX = config('SOCKET_PREFIX', default='/ws')
SOCKET_SOCKETIO_PATH = config('SOCKET_SOCKETIO_PATH', default='/socketio')
