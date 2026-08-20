"""Targeted screenshot source replacement without rewriting the article body."""

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.nodes.image_fetcher import (
    SCREENSHOT_PATTERN,
    _embedded_screenshot_urls,
    canonicalize_screenshot_url,
)
from agent.state import AgentState
from agent.tools.screenshot import preflight_screenshot_url
from agent.tools.search import search


REPLACEMENT_PATTERN = re.compile(
    r"\[REPLACEMENT:\s*(\d+)\s*\|\s*(https?://[^\s|\]]+)\s*\|\s*([^\]]+)\]"
)


def _normalize_url(url: str) -> str:
    return canonicalize_screenshot_url(url)


def _is_usable_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and parsed.path.rstrip("/") != ""


def _topic_query(topic: str) -> str:
    """Reduce a long writing brief to a safe search-sized subject."""
    title = re.search(r'文章用这个标题[“"]([^”"]+)[”"]', topic)
    if title:
        return title.group(1)[:160]
    first_line = next((line.strip() for line in topic.splitlines() if line.strip()), topic)
    return re.sub(r"https?://\S+", "", first_line)[:160].strip()


def _discover_candidates(
    topic: str,
    targets: list[tuple[int, str]],
    used_normalized: set[str],
) -> list[tuple[str, str]]:
    """Search and preflight real pages instead of asking the LLM to invent URLs."""
    subject = _topic_query(topic)
    queries: list[str] = [
        f"{subject} official documentation guides",
        f"{subject} official API reference examples",
        f"{subject} official GitHub README releases",
    ]
    descriptions = [description for _, description in targets if description]
    for offset in range(0, min(len(descriptions), 4), 4):
        focus = " ".join(descriptions[offset : offset + 4])[:240]
        queries.append(f"{subject} {focus} 官方 文档 功能 示例")
    if not queries:
        queries = [f"{subject} 官方 文档 功能 示例"]

    raw: list[tuple[str, str]] = []
    seen: set[str] = set(used_normalized)
    for query in queries[:4]:
        for item in search(query, max_results=10):
            url = str(item.get("url", "")).strip()
            normalized = _normalize_url(url) if url else ""
            if not url or normalized in seen or not _is_usable_url(url):
                continue
            seen.add(normalized)
            raw.append((url, str(item.get("title", "")).strip()))

    if not raw:
        return []
    with ThreadPoolExecutor(max_workers=min(6, len(raw))) as pool:
        checks = list(pool.map(lambda item: preflight_screenshot_url(item[0]), raw))
    valid_items = [item for item, valid in zip(raw, checks) if valid]

    subject_tokens = {
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9.-]{2,}", subject)
        if token.lower() not in {"api", "pro", "the", "and", "with", "for"}
    }

    def score(item: tuple[str, str]) -> int:
        url, title = item
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        searchable = f"{parsed.path} {title}".lower()
        value = 0
        if host.startswith(("docs.", "doc.", "developer.", "developers.", "platform.")):
            value += 3
        if host == "github.com" and len([part for part in parsed.path.split("/") if part]) >= 2:
            value += 3
        if any(part in parsed.path.lower() for part in ("/docs", "/guides", "/manual", "/reference", "/examples", "/releases")):
            value += 2
        if any(token in host for token in subject_tokens):
            value += 5
        elif host == "github.com" and any(token in parsed.path.lower() for token in subject_tokens):
            value += 4
        elif any(token in searchable for token in subject_tokens):
            value += 1
        if "official" in title.lower() or "官方" in title:
            value += 1
        return value

    return sorted(valid_items, key=score, reverse=True)[:30]


def screenshot_refiller_node(state: AgentState) -> dict:
    """Replace only failed screenshot placeholders with new official-page candidates."""
    draft = state["draft"]
    slots = list(SCREENSHOT_PATTERN.finditer(draft))
    retry_number = state.get("screenshot_retry_count", 0) + 1

    if not slots:
        return {
            "screenshot_retry_count": retry_number,
            "log": state.get("log", []) + ["没有可替换的截图位置，无法补图"],
        }

    used_urls = _embedded_screenshot_urls(draft)
    used_urls.update(state.get("screenshot_source_urls", []))
    used_urls.update(state.get("screenshot_attempted_urls", []))
    used_normalized = {_normalize_url(url) for url in used_urls}
    failed_hosts = sorted({urlparse(url).netloc.lower() for url in used_urls if urlparse(url).netloc})
    targets: list[str] = []
    for index, match in enumerate(slots, start=1):
        failed_url = match.group(1).strip()
        description = (match.group(2) or "").strip()
        used_urls.add(failed_url)
        used_normalized.add(_normalize_url(failed_url))
        targets.append(f"{index}. 原 URL：{failed_url}\n   图片用途：{description}")

    discovered = _discover_candidates(
        state["topic"],
        [(index, (match.group(2) or "").strip()) for index, match in enumerate(slots, start=1)],
        used_normalized,
    )
    discovered_text = "\n".join(
        f"- {url} | {title or '官方页面'}" for url, title in discovered
    ) or "（没有通过预检的候选）"

    prompt = f"""你是网页截图的来源补图助手。不要重写、总结或输出文章正文。
文章主题：{state['topic']}

以下截图位置因白屏、失效、验证页、登录页、限流或画面重复而失败。请仅为每个位置选择一个新的、可公开访问的官方功能子页、官方文档页、官方 GitHub 示例/文档页、产品演示页或结果页。

硬性规则：
- 保持每个位置的图片用途，URL 必须和已失败/已使用 URL 不同。
- 同一页面的查询参数或锚点变化仍视为同一页面，禁止用来凑数。
- 不要官网首页、产品入口、登录/注册/认证页；不要新闻、论坛、博客、公众号或其他博主文章。
- 优先展示具体功能、操作、设置、数据结果或实际效果。
- 避免 API JSON 地址、模型社区页面、聚合排行榜和经常触发验证/限流的页面；如果官方文档站不稳定，改用该项目官方 GitHub 仓库中对应的 README、examples 或 docs 子目录页面。
- 每个位置按优先级给出最多 3 个不同候选，系统会预检并选择第一个真实可访问的页面；不要重复使用同一域名下相同路径。
- 只能从下面这批已联网检索且 HTTP 预检通过的候选中选择，禁止自行编造或改写 URL：
{discovered_text}
- 只输出下列格式，每个位置一行，不得输出其他内容：
  [REPLACEMENT: 序号 | https://官方子页URL | 中文具体图注]
  同一序号可以连续输出最多 3 行备选。

待补位置：
{chr(10).join(targets)}

本轮已失败或已使用的主机（除非没有其他官方来源，否则请避开）：
{chr(10).join(failed_hosts)}

禁止再次使用的 URL：
{chr(10).join(sorted(used_urls))}
"""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    selected: dict[int, tuple[str, str]] = {}
    allowed_normalized = {_normalize_url(url) for url, _ in discovered}
    for match in REPLACEMENT_PATTERN.finditer(response.content.strip()):
        index = int(match.group(1))
        url = match.group(2).strip()
        description = match.group(3).strip()
        normalized = _normalize_url(url)
        if index in selected:
            continue
        if (
            index < 1
            or index > len(slots)
            or normalized in used_normalized
            or normalized not in allowed_normalized
            or not _is_usable_url(url)
            or not preflight_screenshot_url(url)
        ):
            continue
        selected[index] = (url, description)
        used_normalized.add(normalized)

    # Deterministic fallback: fill omitted slots from the verified search pool
    # instead of sending invented URLs into another Playwright round.
    available = iter(discovered)
    for index, slot in enumerate(slots, start=1):
        if index in selected:
            continue
        for url, title in available:
            normalized = _normalize_url(url)
            if normalized in used_normalized:
                continue
            selected[index] = (
                url,
                title or (slot.group(2) or "具体官方功能页面").strip(),
            )
            used_normalized.add(normalized)
            break

    refilled_draft = draft
    for index, match in reversed(list(enumerate(slots, start=1))):
        replacement = selected.get(index)
        if not replacement:
            continue
        url, description = replacement
        placeholder = f"[SCREENSHOT: {url}, {description}]"
        refilled_draft = (
            refilled_draft[:match.start()] + placeholder + refilled_draft[match.end():]
        )

    return {
        "draft": refilled_draft,
        "screenshot_retry_count": retry_number,
        "screenshot_retry_note": "",
        "screenshot_attempted_urls": sorted(used_urls),
        "log": state.get("log", []) + [
            f"补图第 {retry_number} 轮：已替换 {len(selected)}/{len(slots)} 个失败截图位置"
        ],
    }
