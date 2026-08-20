import unittest

import requests

from api.aihot import AIHOT_HOT_TOPICS_URL, AIHotError, fetch_hot_topics


class FakeResponse:
    def __init__(self, payload, error=False):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise requests.HTTPError("upstream failed")

    def json(self):
        return self.payload


class AIHotTests(unittest.TestCase):
    def test_fetches_v1_hot_topics_without_credentials(self):
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse({
                "schemaVersion": 1,
                "count": 1,
                "items": [{
                    "rank": 1,
                    "id": "item-1",
                    "title": "热点标题",
                    "source": {"name": "OpenAI"},
                    "links": {
                        "aihot": "https://aihot.virxact.com/items/item-1",
                        "original": "https://example.com/original",
                    },
                    "sourceCount": 3,
                    "latestAt": "2026-08-19T00:00:00Z",
                }],
            })

        result = fetch_hot_topics(fake_get)

        self.assertEqual(captured["url"], AIHOT_HOT_TOPICS_URL)
        self.assertNotIn("Authorization", captured["headers"])
        self.assertNotIn("Cookie", captured["headers"])
        self.assertEqual(result["items"][0]["title"], "热点标题")
        self.assertEqual(result["items"][0]["original_url"], "https://example.com/original")

    def test_rejects_invalid_or_empty_payload(self):
        with self.assertRaises(AIHotError):
            fetch_hot_topics(lambda *_args, **_kwargs: FakeResponse({"items": []}))

    def test_wraps_upstream_http_failure(self):
        with self.assertRaises(AIHotError):
            fetch_hot_topics(lambda *_args, **_kwargs: FakeResponse({}, error=True))


if __name__ == "__main__":
    unittest.main()
