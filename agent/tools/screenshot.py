"""
Playwright 网页截图工具 — 截取指定 URL 的可视区域作为文章插图

用法：
    from agent.tools.screenshot import take_screenshot
    path = take_screenshot("https://example.com", description="Example homepage")
    # → "data/images/screenshot_1711800000.png"

依赖：
    uv pip install playwright
    python -m playwright install chromium
"""

import os
import time
import threading
from dataclasses import dataclass
from functools import wraps
from urllib.parse import urlparse

import requests

from agent.config import API_BASE_URL


BLOCKED_PAGE_MARKERS = (
    "page not found",
    "404 not found",
    "404 page not found",
    "页面不存在",
    "页面未找到",
    "找不到该页面",
    "performing security verification",
    "verify you are human",
    "verifying you are human",
    "checking your browser",
    "just a moment",
    "cloudflare",
    "access denied",
    "captcha",
    "人机验证",
    "安全验证",
    "访问被拒绝",
)
RAW_HTML_CHALLENGE_MARKERS = (
    "cf-turnstile-response",
    "challenge-platform/h/g/turnstile",
    "hcaptcha-response",
    "g-recaptcha-response",
)
INCOMPLETE_PAGE_MARKERS = (
    "文档正在编辑中", "敬请等待", "敬请期待", "内容建设中",
    "暂无内容", "coming soon", "under construction",
)
THIRD_PARTY_CONTENT_HOSTS = (
    "mp.weixin.qq.com", "weixin.qq.com", "zhihu.com", "sohu.com", "163.com",
    "ifeng.com", "eastmoney.com", "sina.com.cn", "toutiao.com", "bilibili.com",
    "xiaohongshu.com", "instagram.com", "woshipm.com", "juejin.cn", "csdn.net",
    "medium.com", "36kr.com", "huxiu.com", "thepaper.cn", "reddit.com",
    "stackoverflow.com", "youtube.com", "youtu.be", "quora.com",
)
FORUM_HOST_PREFIXES = ("community.", "forum.", "forums.")
ENTRY_PATH_MARKERS = ("/login", "/signin", "/sign-in", "/register", "/signup", "/sign-up", "/auth")
# Reject explicit data-file URLs. Paths containing /api/ may still be real
# documentation pages (for example /api/docs/guides), so content-type
# preflight—not the path name—decides whether they are JSON endpoints.
API_PAGE_MARKERS = (".json", ".xml")
SCREENSHOT_ENGINE_VERSION = "screenshot-v27-search-verified-universal-20260819"
_PROBE_SESSION = requests.Session()
_PROBE_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
)
_PREFLIGHT_CACHE_LOCK = threading.Lock()


def _ttl_preflight_cache(func):
    """Cache stable successes longer than transient failures."""
    cache: dict[str, tuple[float, bool]] = {}

    @wraps(func)
    def wrapped(target_url: str) -> bool:
        now = time.monotonic()
        with _PREFLIGHT_CACHE_LOCK:
            cached = cache.get(target_url)
            if cached:
                created_at, value = cached
                ttl = 1800 if value else 60
                if now - created_at < ttl:
                    return value
        value = func(target_url)
        with _PREFLIGHT_CACHE_LOCK:
            cache[target_url] = (time.monotonic(), value)
        return value

    def cache_clear() -> None:
        with _PREFLIGHT_CACHE_LOCK:
            cache.clear()

    wrapped.cache_clear = cache_clear
    return wrapped
LANDING_ACCOUNT_MARKERS = ("登录", "注册", "sign in", "log in", "sign up")
LANDING_CTA_MARKERS = (
    "免费试用", "立即体验", "立即开始", "立即下载", "下载客户端",
    "下载 workbuddy", "get started", "try for free", "download",
)


def _is_blank_screenshot(filename: str) -> bool:
    """Reject all-white and nearly uniform browser canvases after capture."""
    try:
        from PIL import Image, ImageStat

        with Image.open(filename) as image:
            thumbnail = image.convert("RGB").resize((64, 40))
            stat = ImageStat.Stat(thumbnail)
            mean = sum(stat.mean) / 3
            deviation = sum(stat.stddev) / 3
            pixels = list(thumbnail.get_flattened_data())
        near_white_ratio = sum(
            1 for red, green, blue in pixels
            if red >= 245 and green >= 245 and blue >= 245
        ) / len(pixels)
        return (mean >= 248 and near_white_ratio >= 0.97) or deviation <= 1.5
    except (ImportError, OSError, ValueError, ZeroDivisionError):
        return False


def _is_entry_or_third_party_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    path = parsed.path.rstrip("/").lower()
    if any(host == domain or host.endswith(f".{domain}") for domain in THIRD_PARTY_CONTENT_HOSTS):
        return True
    if host.startswith(FORUM_HOST_PREFIXES):
        return True
    return (
        not path
        or any(marker in path for marker in ENTRY_PATH_MARKERS)
        or any(marker in path for marker in API_PAGE_MARKERS)
    )


def _is_generic_landing_page(url: str, visible_text: str) -> bool:
    """Identify product entry/marketing pages that merely have a non-root slug."""
    path_segments = [segment for segment in urlparse(url).path.split("/") if segment]
    if len(path_segments) > 1:
        return False
    has_account_action = any(marker in visible_text for marker in LANDING_ACCOUNT_MARKERS)
    has_conversion_action = any(marker in visible_text for marker in LANDING_CTA_MARKERS)
    return has_account_action and has_conversion_action


@_ttl_preflight_cache
def preflight_screenshot_url(target_url: str) -> bool:
    """Cheaply reject dead/API/challenge URLs before launching Playwright.

    This is deliberately only a candidate filter; Playwright still performs
    the final visual/content validation. A GET fallback is used because many
    documentation CDNs do not implement HEAD correctly.
    """
    if _is_entry_or_third_party_url(target_url):
        return False
    try:
        response = _PROBE_SESSION.get(target_url, timeout=(5, 12), allow_redirects=True)
        if response.status_code >= 400 or response.status_code in {204, 205, 429}:
            return False
        resolved = response.url or target_url
        if _is_entry_or_third_party_url(resolved):
            return False
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml+xml")):
            return False
        sample = response.text[:120_000].lower()
        # Normal sites (notably GitHub) ship dormant captcha component code in
        # their HTML. Only reject explicit challenge widgets here; Playwright
        # checks the actually visible page text below.
        if any(marker in sample for marker in RAW_HTML_CHALLENGE_MARKERS):
            return False
        if any(marker in sample for marker in ("404 not found", "page not found", "页面不存在", "页面未找到")):
            return False
        return len(sample.strip()) >= 500
    except requests.RequestException:
        return False


@dataclass
class ScreenshotResult:
    """截图结果"""
    url: str        # 本地文件路径
    alt: str        # 描述文字
    credit: str     # 来源说明
    source_url: str = ""  # Playwright redirects resolved to this page


def take_screenshot(
    target_url: str,
    description: str = "",
    width: int = 1280,
    height: int = 800,
    clip: dict | None = None,
) -> ScreenshotResult | None:
    """
    使用 Playwright 截取网页可视区域。

    Args:
        target_url:  要截图的 URL
        description: 图片描述（用于 alt 文字）
        width:       视口宽度，默认 1280
        height:      视口高度，默认 800
        clip:        可选裁剪区域 {"x": 0, "y": 0, "width": 1280, "height": 750}

    Returns:
        ScreenshotResult 或 None（失败时）
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [Screenshot] playwright 未安装，请运行: uv pip install playwright && python -m playwright install chromium")
        return None

    os.makedirs("data/images", exist_ok=True)
    basename = f"screenshot_{int(time.time())}_{hash(target_url) % 10000:04d}.png"
    filename = f"data/images/{basename}"

    print(f"  [Screenshot] 截图: {target_url}")

    if _is_entry_or_third_party_url(target_url):
        print(f"  [Screenshot] 已跳过入口页、登录页或第三方内容页: {target_url}")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": width, "height": height})

            # DOMContentLoaded is more reliable for modern docs sites than
            # networkidle, which can wait forever on analytics/websocket calls.
            response = None
            try:
                response = page.goto(target_url, wait_until="domcontentloaded", timeout=20000)
            except Exception:
                response = page.goto(target_url, wait_until="load", timeout=20000)
            # Give client-side renderers enough time to paint without holding
            # every failed candidate for the old 60-second double timeout.
            page.wait_for_timeout(2500)

            # A documentation CDN can briefly answer 429 while the browser
            # pool is warming. One delayed retry recovers transient throttling;
            # persistent 429s are rejected as unusable pages.
            if response is not None and response.status == 429:
                page.wait_for_timeout(3000)
                response = page.goto(target_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(1500)

            # A successful navigation can still display an HTTP error document.
            if response is not None and response.status >= 400:
                print(f"  [Screenshot] 已跳过 HTTP {response.status} 错误页: {target_url}")
                browser.close()
                return None

            # Redirects can land on an entry page even when the original URL was valid.
            if _is_entry_or_third_party_url(page.url):
                print(f"  [Screenshot] 已跳过重定向后的入口页或第三方内容页: {page.url}")
                browser.close()
                return None

            # Do not publish missing-page, anti-bot, login, CAPTCHA or access-denied pages as article images.
            visible_text = (
                f"{page.title()}\n"
                f"{page.locator('body').inner_text(timeout=5000)}"
            ).lower()
            page_html = page.content().lower()
            if any(marker in visible_text for marker in BLOCKED_PAGE_MARKERS) or any(
                marker in page_html for marker in RAW_HTML_CHALLENGE_MARKERS
            ):
                print(f"  [Screenshot] 已跳过验证或访问拦截页: {target_url}")
                browser.close()
                return None
            if any(marker in visible_text for marker in INCOMPLETE_PAGE_MARKERS):
                print(f"  [Screenshot] 已跳过未完成或暂无内容的页面: {target_url}")
                browser.close()
                return None
            if _is_generic_landing_page(page.url, visible_text):
                print(f"  [Screenshot] 已跳过产品入口或营销落地页: {page.url}")
                browser.close()
                return None
            if page.locator("input[type='password']").count() or (
                "登录" in visible_text and page.locator("input").count() >= 2
            ):
                print(f"  [Screenshot] 已跳过登录或认证表单页: {target_url}")
                browser.close()
                return None

            resolved_url = page.url

            # Capture the substantive feature/document area rather than a
            # generic site header. This also makes pages from the same official
            # documentation host visually distinct for deduplication.
            content_locator = page.locator(
                "main, article, [role='main'], .markdown-body, .prose, .content"
            ).first
            try:
                if content_locator.count() and content_locator.bounding_box():
                    content_locator.scroll_into_view_if_needed(timeout=3000)
                    page.wait_for_timeout(500)
                else:
                    page.mouse.wheel(0, 520)
                    page.wait_for_timeout(500)
            except Exception:
                page.mouse.wheel(0, 520)
                page.wait_for_timeout(500)

            # 截图
            screenshot_opts: dict = {"path": filename}
            if clip:
                screenshot_opts["clip"] = clip
            else:
                screenshot_opts["full_page"] = False

            page.screenshot(**screenshot_opts)
            if _is_blank_screenshot(filename):
                # The page may have reached networkidle before its client-side
                # renderer painted. Retry the same URL before replacing it.
                print(f"  [Screenshot] 首次截图为空，等待页面继续渲染后重试: {target_url}")
                page.wait_for_timeout(4000)
                page.screenshot(**screenshot_opts)
            browser.close()

        if _is_blank_screenshot(filename):
            os.remove(filename)
            print(f"  [Screenshot] 已跳过空白或未渲染完成的页面: {target_url}")
            return None

        print(f"  [Screenshot] 保存: {filename}")
        alt = description or target_url
        # 返回 API 可访问的 URL，与 upload-image 端点保持一致
        api_url = f"{API_BASE_URL}/api/images/{basename}"
        return ScreenshotResult(
            url=api_url,
            alt=alt,
            credit=description or target_url,
            source_url=resolved_url,
        )

    except Exception as e:
        print(f"  [Screenshot] 截图失败 ({target_url}): {e}")
        return None
