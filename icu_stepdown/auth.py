import os
from typing import Callable, Tuple


def resolve_expected_credentials(
    secret_get: Callable[[str], str | None] | None = None,
    env: dict | None = None,
) -> Tuple[str, str, bool]:
    """
    Return (username, password, using_demo_defaults).

    If ICU_APP_USER/ICU_APP_PASS are set (env or secrets), they override defaults.
    Demo defaults are for local/testing only. Production credentials must not be hard coded.
    """
    env = env or os.environ
    user = env.get("ICU_APP_USER")
    password = env.get("ICU_APP_PASS")
    if secret_get:
        user = user or secret_get("ICU_APP_USER")
        password = password or secret_get("ICU_APP_PASS")
    if user and password:
        return user, password, False
    # Demo defaults for local use only; override with ICU_APP_USER/ICU_APP_PASS in production.
    return "admin", "Test#12", True


def validate_credentials(username: str, password: str, secret_get: Callable[[str], str | None] | None = None, env: dict | None = None) -> Tuple[bool, bool]:
    expected_user, expected_pass, using_demo = resolve_expected_credentials(secret_get=secret_get, env=env)
    return (username == expected_user and password == expected_pass), using_demo
