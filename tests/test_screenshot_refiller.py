import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.nodes.screenshot_refiller import _discover_candidates, screenshot_refiller_node


class ScreenshotRefillerTests(unittest.TestCase):
    @patch("agent.nodes.screenshot_refiller.preflight_screenshot_url")
    @patch("agent.nodes.screenshot_refiller.search")
    def test_discovery_keeps_only_real_search_results(self, search, preflight):
        search.return_value = [
            {"url": "https://docs.example.com/guides/a", "title": "Official guide A"},
            {"url": "https://docs.example.com/missing", "title": "Missing"},
            {"url": "https://docs.example.com/guides/b", "title": "Official guide B"},
        ]
        preflight.side_effect = lambda url: "missing" not in url
        found = _discover_candidates("Example 产品教程", [(1, "功能")], set())
        self.assertEqual(
            {"https://docs.example.com/guides/a", "https://docs.example.com/guides/b"},
            {url for url, _ in found},
        )

    @patch("agent.nodes.screenshot_refiller.get_llm")
    @patch("agent.nodes.screenshot_refiller._discover_candidates")
    def test_refiller_rejects_invented_url_and_uses_verified_pool(self, discover, get_llm):
        discover.return_value = [
            ("https://docs.example.com/guides/a", "功能 A"),
            ("https://docs.example.com/guides/b", "功能 B"),
        ]
        get_llm.return_value.invoke.return_value = SimpleNamespace(
            content="[REPLACEMENT: 1 | https://invented.example/not-real | 编造页面]"
        )
        state = {
            "topic": "Example 产品教程",
            "draft": (
                "[SCREENSHOT: https://old.example/dead-a, 位置一]\n"
                "[SCREENSHOT: https://old.example/dead-b, 位置二]"
            ),
            "screenshot_attempted_urls": [],
            "screenshot_source_urls": [],
            "log": [],
        }
        result = screenshot_refiller_node(state)
        self.assertNotIn("invented.example", result["draft"])
        self.assertIn("https://docs.example.com/guides/a", result["draft"])
        self.assertIn("https://docs.example.com/guides/b", result["draft"])


if __name__ == "__main__":
    unittest.main()
