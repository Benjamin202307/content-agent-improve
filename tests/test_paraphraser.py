import unittest
from unittest.mock import patch

from agent.nodes.paraphraser import (
    ParaphraseAPIError,
    _build_plan,
    _extract_marked_text_partial,
    _request,
    _rewrite_segments,
    _restore_segments,
    paraphraser_node,
)


def _config(key: str, default: str = "") -> str:
    values = {
        "PARAPHRASE_API_KEY": "test-key",
        "PARAPHRASE_API_URL": "https://example.invalid/paraphrase",
        "PARAPHRASE_PRESET": "aimove_matrix_s_1",
    }
    return values.get(key, default)


class ParaphraserNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.draft = (
            "# 主標題\n\n"
            "## 功能小標題\n\n"
            "這是一段**重點內容**，查看[官方文檔](https://example.com/docs)。\n\n"
            "- 第一個項目\n"
            "- 第二個項目\n\n"
            "[SCREENSHOT: https://example.com/feature, 具體功能效果]\n\n"
            "```python\nprint('繁體程式碼不改寫')\n```\n"
        )

    def test_inline_markup_is_one_request_unit_and_restores_exactly(self):
        draft = "这是**重点**，查看[官方文档](https://example.com/docs)。"
        plan = _build_plan(draft)
        self.assertEqual(1, len(plan.texts))
        self.assertEqual(4, len(plan.protected))
        rewritten = [plan.texts[0].replace("这是", "改写：这是")]
        self.assertEqual(
            "改写：这是**重点**，查看[官方文档](https://example.com/docs)。",
            _restore_segments(plan, rewritten),
        )

    def test_nested_or_leaked_marker_is_treated_as_missing(self):
        response = (
            "CASEG000000START 正常改写 CASEG000000END\n"
            "CASEG000001START 污染文本 CASEG000007START CASEG000001END"
        )
        found, missing = _extract_marked_text_partial(response, [0, 1])

        self.assertEqual({0: "正常改写"}, found)
        self.assertEqual([1], missing)

    def test_short_headings_and_list_items_share_a_semantic_request_unit(self):
        draft = (
            "# AI\n\n"
            "这是足够长的正文，用于验证短标题不会再被拆成独立的三到五字符请求。"
            "它会与后面的段落保持同一个改写单元，并在本地恢复原来的标题和换行格式。\n\n"
            "- 要点\n"
            "- 另一个要点\n"
        )
        plan = _build_plan(draft)

        self.assertEqual(1, len(plan.texts))
        self.assertGreaterEqual(len(plan.texts[0]), 96)
        self.assertIn("AI", plan.texts[0])
        self.assertIn("要点", plan.texts[0])
        self.assertNotIn("\n", plan.texts[0])
        restored = _restore_segments(plan, plan.texts)
        self.assertEqual(draft, restored)

    @patch("agent.nodes.paraphraser.get_config", side_effect=_config)
    @patch("agent.nodes.paraphraser._request")
    def test_rewrites_text_but_preserves_structure(self, request, _get_config):
        request.side_effect = lambda text, *_: (
            text.replace("主標題", "改寫後標題")
            .replace("功能小標題", "改寫後小標題")
            .replace("這是一段", "這是一段已改寫的"),
            "request-test-1234",
        )

        result = paraphraser_node({"draft": self.draft, "log": []})

        article = result["draft"]
        self.assertTrue(result["paraphrase_applied"])
        self.assertIn("# 改写后标题", article)
        self.assertIn("## 改写后小标题", article)
        self.assertIn("**重点内容**", article)
        self.assertIn("[官方文档](https://example.com/docs)", article)
        self.assertIn("- 第一个项目", article)
        self.assertIn("[SCREENSHOT: https://example.com/feature, 具体功能效果]", article)
        self.assertIn("print('繁体程式码不改写')", article)
        self.assertNotIn("CASEG", article)
        self.assertEqual(1, len(result["paraphrase_request_ids"]))
        self.assertEqual(1, result["paraphrase_request_count"])

    @patch("agent.nodes.paraphraser.get_config", side_effect=_config)
    @patch("agent.nodes.paraphraser._request", side_effect=TimeoutError("offline"))
    def test_api_failure_stops_generation(self, _request, _get_config):
        with self.assertRaisesRegex(RuntimeError, "指定 POST 改写失败"):
            paraphraser_node({"draft": self.draft, "log": []})

    @patch("agent.nodes.paraphraser.get_config", side_effect=_config)
    @patch("agent.nodes.paraphraser._request")
    def test_marker_loss_with_empty_plain_retry_stops_generation(self, request, _get_config):
        request.side_effect = [("响应没有结构标记", "request-test-missing")] + [("", "request-test-empty")] * 8
        with self.assertRaisesRegex(RuntimeError, "指定 POST 改写失败"):
            paraphraser_node({"draft": self.draft, "log": []})

    @patch("agent.nodes.paraphraser.get_config", return_value="")
    def test_missing_key_cannot_skip_rewrite(self, _get_config):
        with self.assertRaisesRegex(RuntimeError, "禁止跳过指定 POST 改写"):
            paraphraser_node({"draft": self.draft, "log": []})

    @patch("agent.nodes.paraphraser.get_config", side_effect=_config)
    @patch("agent.nodes.paraphraser._request")
    def test_unchanged_api_response_stops_generation(self, request, _get_config):
        request.side_effect = lambda text, *_: (text, "request-test-unchanged")
        with self.assertRaisesRegex(RuntimeError, "没有产生实际改写"):
            paraphraser_node({"draft": self.draft, "log": []})

    @patch("agent.nodes.paraphraser._request")
    def test_marker_loss_retries_only_failed_batch_as_plain_text(self, request):
        def response(text, *_, **__):
            if "请保持原意并自然改写" in text:
                target = text.split("CASEG000000START ", 1)[1].split(
                    " CASEG000000END", 1
                )[0]
                return (
                    f"CASEG000000START 改写后{target} CASEG000000END\n"
                    "CASEG000001START 上下文已改写 CASEG000001END",
                    f"single-{target}",
                )
            if "CASEG" in text:
                return "服务端改写了批次但未保留标记", "batch-request"
            return f"改写后{text}", f"single-{text}"

        request.side_effect = response
        texts = ["第一段原文", "第二段原文", "第三段原文"]
        rewritten, calls, request_ids, fallback_count = _rewrite_segments(
            texts, "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["改写后第一段原文", "改写后第二段原文", "改写后第三段原文"], rewritten)
        self.assertEqual(4, calls)
        self.assertEqual(3, fallback_count)
        self.assertEqual(4, len(request_ids))

    @patch("agent.nodes.paraphraser._request")
    def test_partial_batch_marker_loss_keeps_successes_and_only_recovers_missing(self, request):
        calls = []

        def response(text, *_, **__):
            calls.append(text)
            if len(calls) == 1:
                return (
                    "\n".join(
                        f"CASEG{index:06d}START 已改写第{index}段 CASEG{index:06d}END"
                        for index in range(7)
                    ),
                    "partial-batch",
                )
            self.assertIn("请保持原意并自然改写", text)
            return "服务端成功响应但删除了短文本边界", "short-without-marker"

        request.side_effect = response
        original = [f"第{index}段原文" for index in range(8)]
        rewritten, count, request_ids, fallback_count = _rewrite_segments(
            original, "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual([f"已改写第{index}段" for index in range(7)], rewritten[:7])
        self.assertEqual(original[7], rewritten[7])
        self.assertEqual(2, count)
        self.assertEqual(["partial-batch", "short-without-marker"], request_ids)
        self.assertEqual(1, fallback_count)

    @patch("agent.nodes.paraphraser._request")
    def test_rich_partial_marker_loss_preserves_only_unframed_short_span(self, request):
        calls = []

        def response(text, *_, **__):
            calls.append(text)
            if len(calls) == 1:
                return (
                    "CASEG000000START 已改写正文 CASEG000000END",
                    "rich-partial",
                )
            self.assertIn("请保持原意并自然改写", text)
            return "服务端删除了短标题边界", "rich-short-unframed"

        request.side_effect = response
        original = "正文内容[CAGKEEP000000]短标题"
        rewritten, count, request_ids, fallback_count = _rewrite_segments(
            [original], "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["已改写正文[CAGKEEP000000]短标题"], rewritten)
        self.assertEqual(2, count)
        self.assertEqual(["rich-partial", "rich-short-unframed"], request_ids)
        self.assertEqual(1, fallback_count)

    @patch("agent.nodes.paraphraser._request")
    def test_protected_placeholder_loss_retries_with_natural_token_variant(self, request):
        original = "正文内容[CAGKEEP000000]"

        def response(text, *_, **__):
            if "CASEG" in text:
                return "CASEG000000START 改写后的正文 CASEG000000END", "batch"
            if "【CAGKEEP000000】" in text:
                return "改写后的正文【CAGKEEP000000】", "variant"
            if text == "正文内容":
                raise RuntimeError("span fallback unavailable")
            return "改写后的正文", "plain"

        request.side_effect = response
        rewritten, calls, request_ids, fallback_count = _rewrite_segments(
            [original], "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["改写后的正文[CAGKEEP000000]"], rewritten)
        self.assertEqual(1, calls)
        self.assertEqual(["batch"], request_ids)
        self.assertEqual(0, fallback_count)

    @patch("agent.nodes.paraphraser._request")
    def test_protected_placeholder_loss_can_rewrite_visible_spans_without_tokens(self, request):
        original = "正文内容[CAGKEEP000000]"

        def response(text, *_, **__):
            if "CASEG" in text:
                return "CASEG000000START 改写后的正文 CASEG000000END", "batch"
            if "CAGKEEP" in text:
                return "接口仍然删除标记", "marked"
            return "改写后的正文内容", "span"

        request.side_effect = response
        rewritten, calls, request_ids, fallback_count = _rewrite_segments(
            [original], "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["改写后的正文[CAGKEEP000000]"], rewritten)
        self.assertEqual(1, calls)
        self.assertEqual(["batch"], request_ids)
        self.assertEqual(0, fallback_count)

    @patch("agent.nodes.paraphraser.time.sleep")
    @patch("agent.nodes.paraphraser.requests.post")
    def test_request_retries_transient_http_500(self, post, sleep):
        class FakeResponse:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self.headers = {}
                self._payload = payload or {}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise AssertionError("transient error should be handled before raise_for_status")

            def json(self):
                return self._payload

        post.side_effect = [
            FakeResponse(500),
            FakeResponse(502),
            FakeResponse(200, {"success": True, "paraphrased_text": "改写后的内容", "request_id": "retry-ok"}),
        ]

        rewritten, request_id = _request(
            "原始内容", "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual("改写后的内容", rewritten)
        self.assertEqual("retry-ok", request_id)
        self.assertEqual(3, post.call_count)
        self.assertEqual(4, sleep.call_count)

    @patch("agent.nodes.paraphraser._request")
    def test_batch_http_500_falls_back_to_plain_text(self, request):
        request.side_effect = [
            ParaphraseAPIError("改写接口暂时不可用", 500),
            (
                "CASEG000000START 改写后的单段文本 CASEG000000END\n"
                "CASEG000001START 上下文已改写 CASEG000001END",
                "single-request",
            ),
        ]

        rewritten, calls, _request_ids, fallback_count = _rewrite_segments(
            ["原始单段文本"], "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["改写后的单段文本"], rewritten)
        self.assertEqual(2, calls)
        self.assertEqual(1, fallback_count)

    @patch("agent.nodes.paraphraser._request")
    def test_plain_fallback_uses_fast_retry_without_outage_wait(self, request):
        calls = []

        def response(text, *args, **kwargs):
            calls.append((text, args, kwargs))
            if "请保持原意并自然改写" in text:
                return (
                    "CASEG000000START 改写后单段 CASEG000000END\n"
                    "CASEG000001START 上下文已改写 CASEG000001END",
                    "single-request",
                )
            if "CASEG" in text:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            return "改写后单段", "single-request"

        request.side_effect = response
        _rewrite_segments(
            ["第一段原文", "第二段原文"],
            "https://example.invalid",
            "aimove_matrix_s_1",
            "test-key",
        )

        fallback_calls = [
            kwargs for text, _args, kwargs in calls if "请保持原意并自然改写" in text
        ]
        self.assertEqual(2, len(fallback_calls))
        for kwargs in fallback_calls:
            self.assertEqual(3, kwargs["attempts"])
            self.assertEqual(0, kwargs["outage_wait_seconds"])

    @patch("agent.nodes.paraphraser._request")
    def test_batch_http_500_falls_back_directly_to_plain_text(self, request):
        def response(text, *_, **__):
            if "请保持原意并自然改写" in text:
                target = text.split("CASEG000000START ", 1)[1].split(
                    " CASEG000000END", 1
                )[0]
                return (
                    f"CASEG000000START {target.replace('原文', '改写后')} CASEG000000END\n"
                    "CASEG000001START 上下文已改写 CASEG000001END",
                    "split-request",
                )
            if "CASEG" in text:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            return text.replace("原文", "改写后"), "split-request"

        request.side_effect = response
        rewritten, calls, _request_ids, fallback_count = _rewrite_segments(
            ["第一段原文", "第二段原文", "第三段原文", "第四段原文"],
            "https://example.invalid",
            "aimove_matrix_s_1",
            "test-key",
        )

        self.assertEqual(
            ["第一段改写后", "第二段改写后", "第三段改写后", "第四段改写后"],
            rewritten,
        )
        self.assertEqual(5, calls)
        self.assertEqual(4, fallback_count)

    @patch("agent.nodes.paraphraser.time.sleep")
    @patch("agent.nodes.paraphraser._request")
    def test_transient_segment_is_recovered_after_cooldown(self, request, sleep):
        plain_calls = 0

        def response(text, *_, **__):
            nonlocal plain_calls
            if "CASEG" in text:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            plain_calls += 1
            if plain_calls == 1:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            return "冷却后改写成功", "recovery-request"

        request.side_effect = response
        rewritten, calls, request_ids, fallback_count = _rewrite_segments(
            ["这是一个足够长的待恢复文本片段，" * 8],
            "https://example.invalid",
            "aimove_matrix_s_1",
            "test-key",
        )

        self.assertEqual(["冷却后改写成功"], rewritten)
        self.assertEqual(3, calls)
        self.assertEqual(["recovery-request"], request_ids)
        self.assertEqual(1, fallback_count)
        sleep.assert_called_once_with(2)

    @patch("agent.nodes.paraphraser.time.sleep")
    @patch("agent.nodes.paraphraser._request")
    def test_persistent_segment_500_has_bounded_recovery(self, request, sleep):
        request.side_effect = ParaphraseAPIError("改写接口暂时不可用", 500)

        with self.assertRaisesRegex(ParaphraseAPIError, "失败片段索引 0"):
            _rewrite_segments(
                ["这是一个足够长且持续失败的文本片段"],
                "https://example.invalid",
                "aimove_matrix_s_1",
                "test-key",
            )

        self.assertEqual([2, 5, 10], [call.args[0] for call in sleep.call_args_list])
        self.assertEqual(5, request.call_count)

    @patch("agent.nodes.paraphraser.time.sleep")
    @patch("agent.nodes.paraphraser._request")
    def test_non_transient_segment_error_is_not_retried(self, request, sleep):
        request.side_effect = [
            ParaphraseAPIError("批次错误", 500),
            ParaphraseAPIError("鉴权失败", 401),
        ]

        with self.assertRaisesRegex(ParaphraseAPIError, "鉴权失败"):
            _rewrite_segments(
                ["这是一个足够长的鉴权失败文本片段"],
                "https://example.invalid",
                "aimove_matrix_s_1",
                "test-key",
            )

        sleep.assert_not_called()
        self.assertEqual(2, request.call_count)

    @patch("agent.nodes.paraphraser._request")
    def test_short_segment_never_uses_raw_request_after_http_500(self, request):
        calls = []

        def response(text, *_, **__):
            calls.append(text)
            if "请保持原意并自然改写" in text:
                return (
                    "CASEG000000START 改写后短文 CASEG000000END\n"
                    "CASEG000001START 上下文已改写 CASEG000001END",
                    "context-request",
                )
            if "CASEG" in text:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            self.fail(f"不应发送短文本裸请求: {text!r}")

        request.side_effect = response
        rewritten, calls, _request_ids, fallback_count = _rewrite_segments(
            ["短文"], "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["改写后短文"], rewritten)
        self.assertEqual(2, calls)
        self.assertEqual(1, fallback_count)

    @patch("agent.nodes.paraphraser._request")
    def test_twenty_two_character_segment_uses_context_wrapper(self, request):
        calls = []

        def response(text, *_, **__):
            calls.append(text)
            if "请保持原意并自然改写" in text:
                return (
                    "CASEG000000START 改写后的二十二字符目标文本 CASEG000000END\n"
                    "CASEG000001START 上下文已改写 CASEG000001END",
                    "context-22",
                )
            if "CASEG" in text:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            self.fail(f"不应发送 22 字符短文本裸请求: {text!r}")

        request.side_effect = response
        rewritten, count, request_ids, fallback_count = _rewrite_segments(
            ["这是一个长度约为二十二字符的标题"],
            "https://example.invalid",
            "aimove_matrix_s_1",
            "test-key",
        )

        self.assertEqual(["改写后的二十二字符目标文本"], rewritten)
        self.assertEqual(2, count)
        self.assertEqual(["context-22"], request_ids)
        self.assertEqual(1, fallback_count)

    @patch("agent.nodes.paraphraser.time.sleep")
    @patch("agent.nodes.paraphraser._request")
    def test_rich_short_span_500_recovers_with_context_wrapper(self, request, sleep):
        original = "七字短标题测试[CAGKEEP000000]"
        calls = []

        def response(text, *_, **__):
            calls.append(text)
            if len(calls) == 1:
                raise ParaphraseAPIError("改写接口暂时不可用", 500)
            self.assertIn("CASEG000001START", text)
            return (
                "CASEG000000START 改写后短标题 CASEG000000END\n"
                "CASEG000001START 上下文已改写 CASEG000001END",
                "context-recovery",
            )

        request.side_effect = response
        rewritten, count, ids, fallback_count = _rewrite_segments(
            [original], "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(["改写后短标题[CAGKEEP000000]"], rewritten)
        self.assertEqual(2, count)
        self.assertEqual(["context-recovery"], ids)
        self.assertEqual(1, fallback_count)
        sleep.assert_not_called()

    @patch("agent.nodes.paraphraser._request")
    def test_eighty_five_segments_are_split_into_small_batches(self, request):
        def response(text, *_, **__):
            return text.replace("原文", "改写后"), "batch-request"

        request.side_effect = response
        texts = [f"第{i}段原文" for i in range(85)]
        rewritten, calls, _request_ids, fallback_count = _rewrite_segments(
            texts, "https://example.invalid", "aimove_matrix_s_1", "test-key"
        )

        self.assertEqual(11, calls)
        self.assertEqual(0, fallback_count)
        self.assertEqual(85, sum(before != after for before, after in zip(texts, rewritten)))


if __name__ == "__main__":
    unittest.main()
