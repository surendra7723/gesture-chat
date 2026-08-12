"""
Development settings for chatapp project.

Inherits all values from base.py and adds development-only overrides.
"""

from .base import *

DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]

# django-extensions / ipython are dev-only tooling. Import them lazily so
# that environments (e.g. CI) that only install base requirements still
# function — they simply won't have shell_plus available.
try:
    import django_extensions  # noqa: F401
    INSTALLED_APPS += ["django_extensions"]
    SHELL_PLUS = "ipython"
except ImportError:
    pass

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}

# Allow all origins during local development.
CORS_ALLOW_ALL_ORIGINS = True
