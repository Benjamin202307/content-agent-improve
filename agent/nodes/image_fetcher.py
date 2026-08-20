import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from agent.state import AgentState
from agent.config import get_config
from agent.tools.unsplash import search_images
from agent.tools.image_gen import generate_image, build_image_prompt, STYLE_PRESETS, PLATFORM_STYLES
from agent.tools.screenshot import take_screenshot, preflight_screenshot_url

# 匹配 [IMAGE: 任意内容]
IMAGE_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")
# 匹配 [SCREENSHOT: url] 或 [SCREENSHOT: url, 描述]
SCREENSHOT_PATTERN = re.compile(r"\[SCREENSHOT:\s*([^\],]+)(?:,\s*([^\]]*))?\]")
PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\s*!\[[^\]]*\]\(prompt-placeholder\)\s*")
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# URL/final-page canonicalization already rejects exact source duplicates.
# Keep the visual threshold strict: documentation pages often share the same
# header/sidebar template, and a loose average-hash threshold incorrectly
# discards genuinely different feature pages.
VISUAL_HASH_DISTANCE_THRESHOLD = 1
MIN_SCREENSHOT_COUNT = 5
MAX_SCREENSHOT_COUNT = 9
# Candidate URLs are intentionally more numerous than the final article
# count. Official sites can be blocked, rate-limited, or require JavaScript;
# keeping a larger pool prevents a few bad candidates from exhausting the run.
MAX_SCREENSHOT_CANDIDATES = 14


def _dedupe_screenshot_urls(draft: str) -> tuple[str, int]:
    """Keep a bounded pool of distinct screenshot URLs and remove repeats."""
    seen_urls: set[str] = set()
    removed = 0

    def keep_first(match: re.Match[str]) -> str:
        nonlocal removed
        url = canonicalize_screenshot_url(match.group(1))
        if url in seen_urls or len(seen_urls) >= MAX_SCREENSHOT_CANDIDATES:
            removed += 1
            return ""
        seen_urls.add(url)
        return match.group(0)

    return SCREENSHOT_PATTERN.sub(keep_first, draft), removed


def canonicalize_screenshot_url(url: str) -> str:
    """Treat query/fragment variants of the same page as one screenshot source."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parsed.path).rstrip("/").lower()
    return f"{parsed.scheme.lower()}://{host}{path}"


def _screenshot_visual_hash(image_url: str) -> int | None:
    """Return a compact perceptual hash for a locally saved Playwright image."""
    try:
        from PIL import Image

        filename = Path(urlparse(image_url).path).name
        path = Path("data/images") / filename
        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((16, 16)).get_flattened_data())
        average = sum(pixels) / len(pixels)
        return sum(1 << index for index, value in enumerate(pixels) if value >= average)
    except (ImportError, OSError, ValueError, ZeroDivisionError):
        return None


def _is_visually_duplicate(fingerprint: int, accepted: list[int]) -> bool:
    """Reject identical and near-identical screenshots, even with different URLs."""
    return any((fingerprint ^ existing).bit_count() <= VISUAL_HASH_DISTANCE_THRESHOLD for existing in accepted)


def _embedded_screenshot_urls(draft: str) -> set[str]:
    """Find the local screenshots already accepted in an earlier refill pass."""
    return {
        match.group(1) for match in MARKDOWN_IMAGE_PATTERN.finditer(draft)
        if "/api/images/screenshot_" in match.group(1)
    }


def _replace_successful_screenshots(draft: str, image_map: dict[str, str]) -> str:
    """Persist only successful captures; leave failed placeholders for targeted refill."""
    result = draft
    for placeholder, replacement in image_map.items():
        if replacement.strip():
            result = result.replace(placeholder, replacement)
    return result


def _get_image_provider() -> str:
    """
    获取图片来源：prompt（仅提示词）、unsplash、ai、screenshot 或 mixed。
    优先读 IMAGE_PROVIDER 配置，未配置则判断是否有 AI 生图的 key 可用。
    screenshot/mixed 模式下 [SCREENSHOT:] 占位符由截图处理，[IMAGE:] 仍走 AI/Unsplash。
    """
    provider = get_config("IMAGE_PROVIDER").lower()
    if provider == "prompt":
        return "prompt"
    if provider == "unsplash":
        return "unsplash"
    if provider in ("screenshot", "mixed"):
        return provider
    if provider in ("openai", "gemini", "openrouter", "replicate", "dashscope"):
        return "ai"
    # 未显式配置：检查是否有 Unsplash key
    if get_config("UNSPLASH_ACCESS_KEY"):
        return "unsplash"
    # 否则尝试 AI
    return "ai"


def _make_prompt_placeholder(keyword: str, style: str | None = None, platform: str | None = None) -> str:
    """生成 AI 绘图提示词占位符"""
    prompt = build_image_prompt(keyword, style, platform)
    return f"\n![{prompt}](prompt-placeholder)\n"


def _make_search_placeholder(keyword: str) -> str:
    """生成搜索关键词占位符（用于 Unsplash 等搜图场景）"""
    return f"\n![{keyword}](prompt-placeholder)\n"


def _fetch_unsplash(placeholder: str, keyword: str, **_) -> tuple[str, str, str]:
    """Unsplash 搜图，失败时生成关键词占位符"""
    imgs = search_images(keyword, count=1)
    if imgs:
        img = imgs[0]
        replacement = f"\n![{img.alt}]({img.url})\n"
        return placeholder, replacement, f"🖼️ 插图：{keyword}"
    replacement = _make_search_placeholder(keyword)
    return placeholder, replacement, f"📋 配图占位（可手动上传）：{keyword}"


def _fetch_ai(placeholder: str, keyword: str, style: str | None = None, platform: str | None = None) -> tuple[str, str, str]:
    """AI 生图，失败时生成绘图提示词占位符"""
    img = generate_image(keyword, style=style, platform=platform)
    if img:
        replacement = f"\n![{img.alt}]({img.url})\n"
        return placeholder, replacement, f"🎨 AI 插图：{keyword}"
    replacement = _make_prompt_placeholder(keyword, style, platform)
    return placeholder, replacement, f"📋 配图占位（可手动上传）：{keyword}"


def _fetch_ai_with_unsplash_fallback(placeholder: str, keyword: str, style: str | None = None, platform: str | None = None) -> tuple[str, str, str]:
    """AI 生图，失败时回退到 Unsplash，都失败则生成绘图提示词占位符"""
    ph, replacement, log = _fetch_ai(placeholder, keyword, style, platform)
    if replacement and "prompt-placeholder" not in replacement:
        return ph, replacement, log
    # 回退 Unsplash
    print(f"  [Fallback] AI 失败，尝试 Unsplash: {keyword}")
    ph2, replacement2, log2 = _fetch_unsplash(placeholder, keyword)
    if replacement2 and "prompt-placeholder" not in replacement2:
        return ph2, replacement2, log2
    # 都失败 → 用 AI 绘图提示词（信息更丰富）
    replacement = _make_prompt_placeholder(keyword, style, platform)
    return placeholder, replacement, f"📋 配图占位（可手动上传）：{keyword}"


def _fetch_screenshot(
    placeholder: str,
    url: str,
    description: str = "",
    **_,
) -> tuple[str, str, str, int | None, str | None]:
    """Playwright 截图，失败时从最终文章中移除该图片位。"""
    if not preflight_screenshot_url(url):
        return placeholder, "", f"截图候选预检失败，已跳过：{url}", None, None
    result = take_screenshot(url, description=description)
    if result:
        replacement = f"\n![{result.alt}]({result.url})\n"
        resolved_source = canonicalize_screenshot_url(result.source_url or url)
        return placeholder, replacement, f"📸 截图：{url}", _screenshot_visual_hash(result.url), resolved_source
    return placeholder, "", f"截图失败，已从文章移除：{url}", None, None


def image_fetcher_node(state: AgentState) -> dict:
    """
    解析初稿中所有 [IMAGE: 关键词] 和 [SCREENSHOT: url, 描述] 占位符，
    根据配置使用 Unsplash 搜图、AI 生图或 Playwright 截图，并发执行。
    """
    print("\n[ImageFetcher] 开始获取插图...")

    draft, duplicate_screenshots = _dedupe_screenshot_urls(state["draft"])
    image_matches = list(IMAGE_PATTERN.finditer(draft))
    screenshot_matches = list(SCREENSHOT_PATTERN.finditer(draft))
    source = _get_image_provider()

    if not image_matches and not screenshot_matches:
        if source == "screenshot":
            existing_count = len(_embedded_screenshot_urls(draft))
            if 5 <= existing_count <= 9:
                return {
                    "images": {},
                    "final_article": draft,
                    "screenshot_success_count": existing_count,
                    "needs_screenshot_retry": False,
                    "log": state.get("log", []) + ["文章截图数量校验通过"],
                }
            return {
                "images": {},
                "final_article": "",
                "screenshot_success_count": 0,
                "needs_screenshot_retry": True,
                "screenshot_retry_note": "模型未提供截图 URL",
                "log": state.get("log", []) + ["模型未提供截图 URL，准备重写，避免交付无图文章"],
            }
        print("  未找到插图占位符，跳过")
        return {
            "images": {},
            "final_article": draft,
            "log": state.get("log", []) + ["🎉 文章生成完成！"],
        }

    # ── 处理 [SCREENSHOT: ...] 占位符 ──
    screenshot_tasks: dict[str, tuple[str, str]] = {}  # placeholder → (url, description)
    for match in screenshot_matches:
        placeholder = match.group(0)
        if placeholder not in screenshot_tasks:
            url = match.group(1).strip()
            desc = (match.group(2) or "").strip()
            screenshot_tasks[placeholder] = (url, desc)
    attempted_urls = set(state.get("screenshot_attempted_urls", []))
    attempted_urls.update(url for url, _ in screenshot_tasks.values())

    # ── 处理 [IMAGE: ...] 占位符（去重）──
    unique: dict[str, str] = {}
    for match in image_matches:
        placeholder = match.group(0)
        if placeholder not in unique:
            unique[placeholder] = match.group(1).strip()

    # 选择图片来源
    platform = state.get("platform")
    style = state.get("image_style") or get_config("IMAGE_STYLE") or PLATFORM_STYLES.get(platform or "", None)

    image_map: dict[str, str] = {}
    screenshot_hashes: dict[str, int] = {}
    resolved_sources: dict[str, str] = {}
    logs: list[str] = []
    if duplicate_screenshots:
        logs.append(f"🖼️ 已跳过 {duplicate_screenshots} 个重复截图")

    # ── 先处理截图任务（始终执行，不受 IMAGE_PROVIDER 影响）──
    if screenshot_tasks:
        print(f"  截图任务：{len(screenshot_tasks)} 个")
        concurrent = get_config("IMAGE_CONCURRENT").lower() in ("true", "1", "yes")
        if concurrent and len(screenshot_tasks) > 1:
            # Two browser sessions are enough to reduce latency while avoiding
            # the 429 responses commonly returned by documentation hosts when
            # several Playwright pages start at once.
            with ThreadPoolExecutor(max_workers=min(len(screenshot_tasks), 2)) as pool:
                futures = {
                    pool.submit(_fetch_screenshot, ph, url, desc): ph
                    for ph, (url, desc) in screenshot_tasks.items()
                }
                for future in as_completed(futures):
                    ph, replacement, log, fingerprint, resolved_source = future.result()
                    image_map[ph] = replacement
                    if fingerprint is not None:
                        screenshot_hashes[ph] = fingerprint
                    if resolved_source:
                        resolved_sources[ph] = resolved_source
                    logs.append(log)
                    print(f"  {log}")
        else:
            for ph, (url, desc) in screenshot_tasks.items():
                _, replacement, log, fingerprint, resolved_source = _fetch_screenshot(ph, url, desc)
                image_map[ph] = replacement
                if fingerprint is not None:
                    screenshot_hashes[ph] = fingerprint
                if resolved_source:
                    resolved_sources[ph] = resolved_source
                logs.append(log)
                print(f"  {log}")

    existing_screenshot_urls = _embedded_screenshot_urls(draft)
    accepted_hashes = [
        fingerprint for url in existing_screenshot_urls
        if (fingerprint := _screenshot_visual_hash(url)) is not None
    ]
    accepted_source_urls = set(state.get("screenshot_source_urls", []))
    for placeholder, (url, _) in screenshot_tasks.items():
        fingerprint = screenshot_hashes.get(placeholder)
        if not image_map.get(placeholder, "").strip() or fingerprint is None:
            continue
        if len(accepted_hashes) >= MAX_SCREENSHOT_COUNT:
            image_map[placeholder] = ""
            logs.append(f"截图已达到 {MAX_SCREENSHOT_COUNT} 张上限，跳过：{url}")
            continue
        resolved_source = resolved_sources.get(placeholder)
        if resolved_source and resolved_source in accepted_source_urls:
            image_map[placeholder] = ""
            logs.append(f"截图最终落到同一页面，已跳过：{url}")
            continue
        if _is_visually_duplicate(fingerprint, accepted_hashes):
            image_map[placeholder] = ""
            logs.append(f"截图画面重复，已跳过：{url}")
            continue
        accepted_hashes.append(fingerprint)
        if resolved_source:
            accepted_source_urls.add(resolved_source)

    successful_screenshot_placeholders = {
        placeholder for placeholder in screenshot_tasks
        if image_map.get(placeholder, "").strip()
    }
    successful_screenshot_count = len(existing_screenshot_urls) + len(successful_screenshot_placeholders)
    screenshot_mode = source == "screenshot"
    if screenshot_mode and not MIN_SCREENSHOT_COUNT <= successful_screenshot_count <= MAX_SCREENSHOT_COUNT:
        failed_urls = [
            url for placeholder, (url, _) in screenshot_tasks.items()
            if placeholder not in successful_screenshot_placeholders
        ]
        logs.append(
            f"截图实际成功 {successful_screenshot_count} 张，未达到最终文章至少 5 张的要求，准备重试"
        )
        return {
            "images": image_map,
            "draft": _replace_successful_screenshots(draft, image_map),
            "final_article": "",
            "screenshot_success_count": successful_screenshot_count,
            "needs_screenshot_retry": True,
            "screenshot_retry_note": "\n".join(failed_urls),
            "screenshot_source_urls": sorted(accepted_source_urls),
            "screenshot_attempted_urls": sorted(attempted_urls),
            "log": state.get("log", []) + logs,
        }

    # ── 再处理 IMAGE 任务 ──
    if unique:
        if source == "prompt":
            # 仅生成提示词占位符，不调用任何 API
            print("  图片来源：仅提示词（用户手动生图后上传）")
            for ph, kw in unique.items():
                replacement = _make_prompt_placeholder(kw, style, platform)
                image_map[ph] = replacement
                logs.append(f"📋 配图提示词：{kw}")
                print(f"  📋 配图提示词：{kw}")
        else:
            if source == "unsplash":
                fetch_fn = _fetch_unsplash
                print("  图片来源：Unsplash")
            else:
                fetch_fn = _fetch_ai_with_unsplash_fallback
                provider = get_config("IMAGE_PROVIDER") or "auto"
                style_label = STYLE_PRESETS.get(style or "", {}).get("label", style or "auto")
                print(f"  图片来源：AI 生图 (provider={provider}, style={style_label})")

            # 并发控制：IMAGE_CONCURRENT=true 时启用并发，默认串行（多数生图模型不支持并发）
            concurrent = get_config("IMAGE_CONCURRENT").lower() in ("true", "1", "yes")

            if concurrent:
                max_workers = 3 if source == "ai" else min(len(unique), 5)
                print(f"  并发模式：max_workers={max_workers}")
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(fetch_fn, ph, kw, style=style, platform=platform): ph
                        for ph, kw in unique.items()
                    }
                    for future in as_completed(futures):
                        ph, replacement, log = future.result()
                        image_map[ph] = replacement
                        logs.append(log)
                        print(f"  {log}")
            else:
                for ph, kw in unique.items():
                    _, replacement, log = fetch_fn(ph, kw, style=style, platform=platform)
                    image_map[ph] = replacement
                    logs.append(log)
                    print(f"  {log}")

    # 替换占位符
    final_article = draft
    for placeholder, replacement in image_map.items():
        final_article = final_article.replace(placeholder, replacement)
    # Never expose failed image jobs as manual-upload cards in the final article.
    final_article = PROMPT_PLACEHOLDER_PATTERN.sub("\n", final_article)

    print(f"  完成，共处理 {len(image_map)} 张插图（截图 {len(screenshot_tasks)}，配图 {len(unique)}）")

    return {
        "images": image_map,
        "final_article": final_article,
        "screenshot_success_count": successful_screenshot_count,
        "screenshot_source_urls": sorted(accepted_source_urls),
        "screenshot_attempted_urls": sorted(attempted_urls),
        "needs_screenshot_retry": False,
        "log": state.get("log", []) + logs + ["🎉 文章生成完成！"],
    }
