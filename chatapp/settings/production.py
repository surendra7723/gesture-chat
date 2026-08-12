"""
Production settings for chatapp project.

Inherits all values from base.py and applies production-safe overrides.
Production deployments set DJANGO_SETTINGS_MODULE=chatapp.settings.production
and provide the required environment variables (e.g. DJANGO_SECRET_KEY).
"""

import os

from .base import *

DEBUG = False

ALLOWED_HOSTS = [
    host for host in os.environ.get("ALLOWED_HOSTS", "").split(",") if host
]

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")],
        },
    },
}

# SECURITY: never allow all origins in production.
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = [
    origin for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if origin
]

# SECRET_KEY must be provided via the environment in production.
SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

CSRF_TRUSTED_ORIGINS = [
    origin
    for origin in os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin
]

# HTTPS / secure cookie enforcement in production.
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
