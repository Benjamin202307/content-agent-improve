import unittest

from agent.publish.wechat_html import md_to_wechat_html


class WeChatHtmlTests(unittest.TestCase):
    def test_plain_text_blocks_stay_light_even_with_dark_code_theme(self):
        markdown = """# 歌词示例

```text
[Intro]
Instrumental intro, clean electric guitar and soft piano
```
"""
        html = md_to_wechat_html(markdown, code_theme="atom-one-dark")["html"]
        self.assertIn("background: #f8f8f8", html.lower())
        self.assertNotIn("background: #282c34", html.lower())

    def test_default_code_theme_is_light(self):
        html = md_to_wechat_html("# 示例\n\n```text\n歌词\n```")["html"]
        self.assertIn("background: #f8f8f8", html.lower())


if __name__ == "__main__":
    unittest.main()
