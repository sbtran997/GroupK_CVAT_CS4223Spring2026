from .base import *

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": "/tmp/groupk_test.db",
    },
}

TEST_RUNNER = "django.test.runner.DiscoverRunner"

for config in RQ_QUEUES.values():
    config["ASYNC"] = False

INSTALLED_APPS = [app for app in INSTALLED_APPS if app not in ["silk", "django_extensions"]]
MIDDLEWARE = [m for m in MIDDLEWARE if "silk" not in m]

PASSWORD_HASHERS = ("django.contrib.auth.hashers.MD5PasswordHasher",)
