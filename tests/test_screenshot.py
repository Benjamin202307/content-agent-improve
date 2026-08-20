import unittest
from unittest.mock import patch

from agent.tools.screenshot import _is_entry_or_third_party_url, preflight_screenshot_url


class ScreenshotCandidateTests(unittest.TestCase):
    def test_api_and_auth_pages_are_rejected_before_playwright(self):
        self.assertTrue(_is_entry_or_third_party_url("https://example.com/data.json"))
        self.assertFalse(_is_entry_or_third_party_url("https://example.com/api/docs/guides"))
        self.assertTrue(_is_entry_or_third_party_url("https://example.com/docs/login"))

    @patch("agent.tools.screenshot._PROBE_SESSION.get")
    def test_preflight_rejects_404_and_accepts_real_html(self, get):
        class Response:
            def __init__(self, status, text, url="https://example.com/docs/feature"):
                self.status_code = status
                self.text = text
                self.url = url
                self.headers = {"content-type": "text/html; charset=utf-8"}

        get.return_value = Response(404, "<html>404 not found</html>")
        preflight_screenshot_url.cache_clear()
        self.assertFalse(preflight_screenshot_url("https://example.com/docs/feature"))
        get.return_value = Response(200, "<html>" + ("feature detail " * 100) + "</html>")
        preflight_screenshot_url.cache_clear()
        self.assertTrue(preflight_screenshot_url("https://example.com/docs/feature"))


if __name__ == "__main__":
    unittest.main()
