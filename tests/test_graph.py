import unittest
from unittest.mock import patch

from agent.graph import save_memory_node


class SaveMemoryNodeTests(unittest.TestCase):
    @patch("agent.graph.save_to_memory", side_effect=RuntimeError("embedding unavailable"))
    def test_optional_embedding_failure_does_not_fail_article(self, _save):
        result = save_memory_node(
            {
                "topic": "测试主题",
                "context": "测试素材",
                "platform": "wechat",
                "final_article": "已完成文章",
                "log": ["文章已完成"],
            }
        )

        self.assertIn("文章已正常生成", result["log"][-1])


if __name__ == "__main__":
    unittest.main()
