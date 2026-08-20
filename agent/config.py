"""
统一配置读取 — 默认优先级：环境变量 > SQLite settings 表 > default。
改写接口密钥由 .env、SQLite 和设置页双向同步维护。

用法：
    from agent.config import get_config
    api_key = get_config("LLM_API_KEY")
"""

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv, set_key
from agent.db import get_setting


# Keep every entry point (uvicorn, CLI, tests and workers) on the same
# configuration source. Existing process environment variables still win
# because python-dotenv does not override them by default.
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8918").rstrip("/")
ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"
PARAPHRASE_KEY_NAME = "PARAPHRASE_API_KEY"


def _read_dotenv_value(key: str) -> str | None:
    """Read the current project .env file, including changes after startup."""
    try:
        if not ENV_FILE_PATH.exists():
            return None
        values = dotenv_values(ENV_FILE_PATH)
        if key not in values:
            return ""
        value = values.get(key)
        return "" if value is None else str(value).strip()
    except OSError:
        return None


def sync_paraphrase_api_key(value: str | None = None) -> str:
    """Keep the editable paraphrase credential identical in .env, DB and env.

    With no explicit value, the file is treated as the external source of
    truth, which makes manual .env edits visible to an already-running server.
    With an explicit value, all three stores are updated for a frontend save.
    """
    if value is None:
        file_value = _read_dotenv_value(PARAPHRASE_KEY_NAME)
        if file_value is not None:
            value = file_value
        else:
            value = os.getenv(PARAPHRASE_KEY_NAME, "").strip()
    value = (value or "").strip()

    # Persist the same value in every configuration store. set_key preserves
    # unrelated .env entries and safely quotes values when needed.
    if _read_dotenv_value(PARAPHRASE_KEY_NAME) != value:
        set_key(str(ENV_FILE_PATH), PARAPHRASE_KEY_NAME, value, quote_mode="auto")
    if os.getenv(PARAPHRASE_KEY_NAME, "") != value:
        os.environ[PARAPHRASE_KEY_NAME] = value
    from agent import db

    if db.get_setting(PARAPHRASE_KEY_NAME, "") != value:
        db.save_settings({PARAPHRASE_KEY_NAME: value})
    return value


def get_config(key: str, default: str = "") -> str:
    """
    读取配置值。普通配置优先级为环境变量 > SQLite settings 表 > default；
    PARAPHRASE_API_KEY 由 .env、SQLite 和设置页双向同步维护。
    环境变量中的空字符串视为未配置，继续回退到 SQLite。
    """
    if key == PARAPHRASE_KEY_NAME:
        return sync_paraphrase_api_key()

    db_val = get_setting(key)
    # The paraphrase credential is editable from the local Settings UI. Once
    # a user saves a value there, it must override the bootstrap .env value;
    # otherwise the UI appears to save successfully while the rewrite node
    # silently keeps using the old environment key.
    env_val = os.getenv(key, "").strip()
    if env_val:
        return env_val
    if db_val:
        return db_val
    return default
