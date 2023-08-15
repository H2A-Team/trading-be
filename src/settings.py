from decouple import config, Csv

# general
DEBUG = config('DEBUG', default=False, cast=bool)

# server
SERVER_HOST = config('SERVER_HOST', default='localhost', cast=str)
SERVER_PORT = config('SERVER_PORT', default=8000, cast=int)

# logging
LOGGING_LEVEL = config('LOGGING_LEVEL', default='info', cast=str)

# cors
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default="*", cast=Csv())
CORS_ALLOWED_METHODS = config('CORS_ALLOWED_METHODS', default="*", cast=Csv())

# rest api
REST_API_PREFIX = config('REST_API_PREFIX', default='/api', cast=str)
REST_API_SWAGGER_URL = config('REST_API_SWAGGER_URL', default='/docs', cast=str)
REST_API_REDOC_URL = config('REST_API_REDOC_URL', default='/redoc', cast=str)

# socket
SOCKET_PREFIX = config('SOCKET_PREFIX', default='/ws', cast=str)

# print('DEBUG', DEBUG)
# print('SERVER_HOST', SERVER_HOST)
# print('SERVER_PORT', SERVER_PORT)
# print('LOGGING_LEVEL', LOGGING_LEVEL)
# print('CORS_ALLOWED_ORIGINS', CORS_ALLOWED_ORIGINS)
# print('CORS_ALLOWED_METHODS', CORS_ALLOWED_METHODS)
# print('REST_API_PREFIX', REST_API_PREFIX)
# print('REST_API_SWAGGER_URL', REST_API_SWAGGER_URL)
# print('REST_API_REDOC_URL', REST_API_REDOC_URL)
# print('SOCKET_PREFIX', SOCKET_PREFIX)
