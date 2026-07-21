from __future__ import annotations

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests

from kufar_server_finder.config import GeminiConfig
from kufar_server_finder.gemini import GeminiAnalyzer


class RateLimitError(Exception):
    status_code = 429


class CallableRateLimitError(Exception):
    def code(self):
        return 429


class FakeModels:
    def __init__(self, key, handler):
        self.key = key
        self.handler = handler

    def generate_content(self, **kwargs):
        return SimpleNamespace(text=self.handler(self.key, kwargs))


class FakeClient:
    def __init__(self, key, handler):
        self.models = FakeModels(key, handler)


class FakeImageResponse:
    def __init__(self, content_type="image/jpeg", content=b"image"):
        self.headers = {"Content-Type": content_type}
        self.content = content

    def raise_for_status(self):
        return None


class FakeImageSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def make_config(**changes):
    config = GeminiConfig(
        api_key="key-1",
        backup_api_keys=tuple(f"key-{index}" for index in range(2, 10)),
        chunk_size=1,
        specs_chunk_size=1,
        request_delay=0,
        max_retries=2,
        vision_max_images=5,
        image_timeout=3,
    )
    return replace(config, **changes)


def parse_chunk_link(contents):
    payload = json.loads(contents.split(":\n", 1)[1])
    return payload[0]["link"]


def analysis_json(link):
    return json.dumps(
        [
            {
                "link": link,
                "is_target": True,
                "is_working": True,
                "real_price": 10,
            }
        ]
    )


def test_creates_three_workers_with_fixed_key_groups():
    created = []

    def factory(key):
        created.append(key)
        return FakeClient(key, lambda key, kwargs: "[]")

    analyzer = GeminiAnalyzer(make_config(), client_factory=factory)

    assert created == [f"key-{index}" for index in range(1, 10)]
    assert analyzer.worker_api_key_groups == (
        ("key-1", "key-2", "key-3"),
        ("key-4", "key-5", "key-6"),
        ("key-7", "key-8", "key-9"),
    )


def test_three_workers_execute_chunks_concurrently_and_keep_result_order():
    barrier = threading.Barrier(3)
    calls = []
    lock = threading.Lock()

    def handler(key, kwargs):
        link = parse_chunk_link(kwargs["contents"])
        with lock:
            calls.append((key, link, threading.current_thread().name))
        barrier.wait(timeout=2)
        return analysis_json(link)

    analyzer = GeminiAnalyzer(
        make_config(),
        client_factory=lambda key: FakeClient(key, handler),
    )
    ads = [{"link": f"link-{index}"} for index in range(1, 4)]

    result = analyzer.analyze_ads(ads)

    assert [item.link for item in result] == ["link-1", "link-2", "link-3"]
    assert {(key, link) for key, link, _ in calls} == {
        ("key-1", "link-1"),
        ("key-4", "link-2"),
        ("key-7", "link-3"),
    }
    assert len({thread_name for _, _, thread_name in calls}) == 3


def test_worker_rotates_only_inside_its_own_key_group_on_429():
    calls = []

    def handler(key, kwargs):
        calls.append(key)
        if key == "key-1":
            raise RateLimitError("429")
        return analysis_json(parse_chunk_link(kwargs["contents"]))

    analyzer = GeminiAnalyzer(
        make_config(),
        client_factory=lambda key: FakeClient(key, handler),
    )

    result = analyzer.analyze_ads([{"link": "x"}])

    assert [item.link for item in result] == ["x"]
    assert calls == ["key-1", "key-2"]
    assert analyzer.client.models.key == "key-2"


def test_worker_wraps_from_third_key_back_to_first_key():
    calls = []

    def handler(key, kwargs):
        calls.append(key)
        if len(calls) <= 3:
            raise RateLimitError("RESOURCE_EXHAUSTED")
        return analysis_json(parse_chunk_link(kwargs["contents"]))

    analyzer = GeminiAnalyzer(
        make_config(max_retries=2),
        client_factory=lambda key: FakeClient(key, handler),
    )

    result = analyzer.analyze_ads([{"link": "x"}])

    assert result[0].link == "x"
    assert calls == ["key-1", "key-2", "key-3", "key-1"]


def test_second_worker_rotates_keys_four_to_six_independently():
    calls = []

    def handler(key, kwargs):
        link = parse_chunk_link(kwargs["contents"])
        calls.append((key, link))
        if key == "key-4":
            raise RateLimitError("429")
        return analysis_json(link)

    analyzer = GeminiAnalyzer(
        make_config(),
        client_factory=lambda key: FakeClient(key, handler),
    )
    result = analyzer.analyze_ads([{"link": "a"}, {"link": "b"}])

    assert [item.link for item in result] == ["a", "b"]
    assert ("key-4", "b") in calls
    assert ("key-5", "b") in calls
    assert not any(key in {"key-1", "key-2", "key-3"} and link == "b" for key, link in calls)


def test_non_rate_limit_errors_retry_same_key(monkeypatch):
    calls = []
    sleeps = []

    def handler(key, kwargs):
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return analysis_json(parse_chunk_link(kwargs["contents"]))

    monkeypatch.setattr("kufar_server_finder.gemini.time.sleep", sleeps.append)
    analyzer = GeminiAnalyzer(
        make_config(max_retries=2),
        client_factory=lambda key: FakeClient(key, handler),
    )

    result = analyzer.analyze_ads([{"link": "x"}])

    assert result[0].link == "x"
    assert calls == ["key-1", "key-1"]
    assert sleeps == [1]


def test_rate_limit_exhaustion_returns_empty_chunk():
    calls = []

    def handler(key, kwargs):
        calls.append(key)
        raise RateLimitError("429")

    analyzer = GeminiAnalyzer(
        make_config(max_retries=1),
        client_factory=lambda key: FakeClient(key, handler),
    )

    assert analyzer.analyze_ads([{"link": "x"}]) == []
    assert calls == ["key-1", "key-2", "key-3"]


def test_invalid_or_empty_response_is_retried_then_skipped():
    responses = iter(["", "not-json"])

    def handler(key, kwargs):
        return next(responses)

    analyzer = GeminiAnalyzer(
        make_config(max_retries=2),
        client_factory=lambda key: FakeClient(key, handler),
    )

    assert analyzer.analyze_ads([{"link": "x"}]) == []


def test_rate_limit_detection_supports_attributes_callable_and_message():
    assert GeminiAnalyzer._is_rate_limit_error(RateLimitError())
    assert GeminiAnalyzer._is_rate_limit_error(CallableRateLimitError())
    assert GeminiAnalyzer._is_rate_limit_error(Exception("resource_exhausted"))
    assert not GeminiAnalyzer._is_rate_limit_error(Exception("bad request"))


def test_all_ai_entrypoints_use_structured_processing():
    def handler(key, kwargs):
        contents = kwargs["contents"]
        link = parse_chunk_link(contents)
        if contents.startswith("Извлеки"):
            return json.dumps([{"link": link, "cpu_model": "Intel Core i5-3470"}])
        if contents.startswith("Нормализуй"):
            return json.dumps(
                [{"link": link, "normalized_cpu_model": "Intel Core i5-3470"}]
            )
        return analysis_json(link)

    analyzer = GeminiAnalyzer(
        make_config(),
        client_factory=lambda key: FakeClient(key, handler),
    )
    ads = [{"link": "x", "description": "d" * 1000}]

    assert analyzer.extract_explicit_specs(ads)[0].cpu_model == "Intel Core i5-3470"
    assert analyzer.infer_specs(ads)[0].cpu_model == "Intel Core i5-3470"
    assert (
        analyzer.normalize_cpu_names(
            [{"link": "x", "cpu_model": "core i5 3470"}]
        )[0].normalized_cpu_model
        == "Intel Core i5-3470"
    )


def test_vision_tasks_run_on_three_workers_concurrently(monkeypatch):
    barrier = threading.Barrier(3)
    used_keys = []
    lock = threading.Lock()

    def handler(key, kwargs):
        prompt = kwargs["contents"][0]
        payload = json.loads(prompt.split("Входные данные: ", 1)[1])
        with lock:
            used_keys.append(key)
        barrier.wait(timeout=2)
        return json.dumps(
            {
                "link": payload["link"],
                "product_type": "other",
                "product_type_confidence": "low",
            }
        )

    analyzer = GeminiAnalyzer(
        make_config(),
        client_factory=lambda key: FakeClient(key, handler),
    )
    monkeypatch.setattr(
        analyzer,
        "_download_image_parts",
        lambda ad, *, image_session=None: [object()],
    )

    result = analyzer.infer_specs_from_images(
        [{"link": f"link-{index}", "images": ["https://image"]} for index in range(3)]
    )

    assert [item.link for item in result] == ["link-0", "link-1", "link-2"]
    assert set(used_keys) == {"key-1", "key-4", "key-7"}


def test_vision_skips_ads_without_downloadable_images(monkeypatch):
    analyzer = GeminiAnalyzer(
        make_config(),
        client_factory=lambda key: FakeClient(
            key, lambda key, kwargs: pytest.fail("AI request must not run")
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_download_image_parts",
        lambda ad, *, image_session=None: [],
    )

    assert analyzer.infer_specs_from_images([{"link": "x"}]) == []


def test_download_image_parts_filters_urls_formats_and_network_errors():
    session = FakeImageSession(
        [
            FakeImageResponse("image/jpeg; charset=binary", b"one"),
            FakeImageResponse("text/html", b"bad"),
            requests.RequestException("network"),
        ]
    )
    analyzer = GeminiAnalyzer(
        make_config(),
        client=FakeClient("single", lambda key, kwargs: "[]"),
        image_session=session,
    )

    parts = analyzer._download_image_parts(
        {
            "images": [
                "invalid",
                "https://example/1.jpg",
                "https://example/2.jpg",
                "https://example/3.jpg",
            ]
        }
    )

    assert len(parts) == 1
    assert session.calls == [
        ("https://example/1.jpg", 3),
        ("https://example/2.jpg", 3),
        ("https://example/3.jpg", 3),
    ]
    assert "User-Agent" in session.headers


def test_payloads_trim_descriptions_and_parallel_fallback_handles_exception():
    analyzer = GeminiAnalyzer(
        make_config(max_description_chars=4),
        client=FakeClient("single", lambda key, kwargs: "[]"),
    )
    ad = {
        "link": "x",
        "title": "PC",
        "price": 5,
        "description": "123456789",
        "characteristics": None,
    }

    assert analyzer._analysis_payload(ad)["description"] == "1234"
    assert analyzer._specs_payload(ad)["description"] == "123456789"
    assert analyzer._run_parallel(
        [1],
        operation=lambda worker, task: (_ for _ in ()).throw(RuntimeError("boom")),
        fallback=lambda: 99,
    ) == [99]
