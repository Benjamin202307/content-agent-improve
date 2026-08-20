from tavily import TavilyClient
from agent.config import get_config
import time

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = get_config("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未配置 TAVILY_API_KEY，请在设置中填写")
        _client = TavilyClient(api_key=api_key)
    return _client


def search(keyword: str, max_results: int = 5) -> list[dict]:
    """
    搜索关键词，返回结构化结果列表。
    每条结果包含 title、content、url 字段。
    """
    # Tavily requests can be reset by local/proxy network paths. Retry briefly,
    # then degrade to an empty result so the LLM can continue without live search.
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = _get_client().search(keyword, max_results=max_results)
            return response.get("results", [])
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    print(f"  [Search] Tavily 暂时不可用，已跳过：{keyword} ({last_error})")
    return []
