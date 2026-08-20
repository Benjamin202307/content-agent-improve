import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.publish.wechat_api import _resolve_local_path, _upload_body_images


class WeChatBodyImageTests(unittest.TestCase):
    def test_resolves_loopback_and_relative_image_urls(self):
        image_dir = Path(__file__).resolve().parents[1] / "data" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "wechat-path-test.png"
        image_path.write_bytes(b"test")
        try:
            for url in (
                "http://127.0.0.1:8918/api/images/wechat-path-test.png",
                "http://localhost:8918/api/images/wechat-path-test.png",
                "/api/images/wechat-path-test.png",
            ):
                self.assertEqual(os.path.normcase(str(image_path)), os.path.normcase(_resolve_local_path(url)))
            self.assertIsNone(_resolve_local_path("https://example.com/api/images/wechat-path-test.png"))
        finally:
            image_path.unlink(missing_ok=True)

    @patch("agent.publish.wechat_api.upload_body_image_from_url")
    def test_upload_replaces_only_src_and_preserves_image_markup(self, upload):
        upload.return_value = "https://mmbiz.qpic.cn/example/0"
        html = (
            '<figure><img alt="具体效果" '
            'src="http://127.0.0.1:8918/api/images/example.png" '
            'style="max-width:100%;border-radius:6px;">'
            '<figcaption>具体效果说明</figcaption></figure>'
        )

        rewritten, count = _upload_body_images("token", html)

        self.assertEqual(1, count)
        self.assertIn('src="https://mmbiz.qpic.cn/example/0"', rewritten)
        self.assertIn('alt="具体效果"', rewritten)
        self.assertIn('style="max-width:100%;border-radius:6px;"', rewritten)
        self.assertIn("<figcaption>具体效果说明</figcaption>", rewritten)

    @patch("agent.publish.wechat_api.upload_body_image_from_url", side_effect=RuntimeError("upstream failed"))
    def test_upload_failure_stops_missing_image_draft(self, _upload):
        html = '<img src="http://127.0.0.1:8918/api/images/example.png">'
        with self.assertRaisesRegex(RuntimeError, "已停止创建缺图草稿"):
            _upload_body_images("token", html)


if __name__ == "__main__":
    unittest.main()
