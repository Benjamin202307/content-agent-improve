"""AIHot v1 匿名只读热点榜客户端。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

AIHOT_HOT_TOPICS_URL = "https://aihot.virxact.com/api/v1/hot-topics"
AIHOT_USER_AGENT = (
    "content-agent-improve/1.0 "
    "(+https://github.com/Benjamin202307/content-agent-improve)"
)


class AIHotError(RuntimeError):
    """AIHot 请求失败或返回结构不符合公开 v1 契约。"""


def fetch_hot_topics(
    http_get: Callable[..., requests.Response] | None = None,
) -> dict[str, Any]:
    """获取并收敛 AIHot Top 10 字段，避免向前端透传未知内容。"""
    get = http_get or requests.get
    try:
        response = get(
            AIHOT_HOT_TOPICS_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": AIHOT_USER_AGENT,
            },
            timeout=(5, 15),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise AIHotError("AIHot 热点榜暂时不可用") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise AIHotError("AIHot 热点榜返回结构异常")

    topics: list[dict[str, Any]] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        links = item.get("links")
        source = item.get("source")
        if not isinstance(links, dict) or not isinstance(source, dict):
            continue

        title = item.get("title")
        original_url = links.get("original")
        aihot_url = links.get("aihot")
        if not all(isinstance(value, str) and value for value in (title, original_url, aihot_url)):
            continue

        topics.append(
            {
                "id": str(item.get("id", "")),
                "rank": item.get("rank"),
                "title": title,
                "source": str(source.get("name", "")),
                "original_url": original_url,
                "aihot_url": aihot_url,
                "source_count": item.get("sourceCount", 0),
                "latest_at": item.get("latestAt"),
            }
        )

    if not topics:
        raise AIHotError("AIHot 热点榜当前没有可用数据")

    return {
        "count": len(topics),
        "items": topics,
        "source": "AIHot",
        "canonical": "https://aihot.virxact.com/hot",
    }
