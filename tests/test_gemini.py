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
        worker_count=3,
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


def test_creates_configured_workers_with_one_shared_key():
    created = []

    def factory(key):
        created.append(key)
        return FakeClient(key, lambda key, kwargs: "[]")

    analyzer = GeminiAnalyzer(
        make_config(worker_count=4),
        client_factory=factory,
    )

    assert created == ["key-1"] * 4
    assert analyzer.worker_count == 4
    assert analyzer.worker_api_keys == ("key-1",) * 4


def test_client_receives_optional_base_url_and_api_version(monkeypatch):
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return FakeClient(kwargs["api_key"], lambda key, request: "[]")

    monkeypatch.setattr("kufar_server_finder.gemini.genai.Client", fake_client)
    analyzer = GeminiAnalyzer(
        make_config(
            worker_count=2,
            base_url="https://proxy.example",
            api_version="v1",
        )
    )

    assert analyzer.worker_count == 2
    assert len(calls) == 2
    assert all(call["api_key"] == "key-1" for call in calls)
    assert all(call["http_options"].base_url == "https://proxy.example" for call in calls)
    assert all(call["http_options"].api_version == "v1" for call in calls)


def test_default_client_omits_http_options(monkeypatch):
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return FakeClient(kwargs["api_key"], lambda key, request: "[]")

    monkeypatch.setattr("kufar_server_finder.gemini.genai.Client", fake_client)
    GeminiAnalyzer(make_config(worker_count=1))

    assert calls == [{"api_key": "key-1"}]


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
        ("key-1", "link-2"),
        ("key-1", "link-3"),
    }
    assert len({thread_name for _, _, thread_name in calls}) == 3


def test_rate_limit_retries_same_single_key(monkeypatch):
    calls = []
    sleeps = []

    def handler(key, kwargs):
        calls.append(key)
        if len(calls) == 1:
            raise RateLimitError("429")
        return analysis_json(parse_chunk_link(kwargs["contents"]))

    monkeypatch.setattr("kufar_server_finder.gemini.time.sleep", sleeps.append)
    analyzer = GeminiAnalyzer(
        make_config(max_retries=2),
        client_factory=lambda key: FakeClient(key, handler),
    )

    result = analyzer.analyze_ads([{"link": "x"}])

    assert [item.link for item in result] == ["x"]
    assert calls == ["key-1", "key-1"]
    assert sleeps == [1]
    assert analyzer.client.models.key == "key-1"


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


def test_rate_limit_exhaustion_returns_empty_chunk(monkeypatch):
    calls = []
    sleeps = []

    def handler(key, kwargs):
        calls.append(key)
        raise RateLimitError("429")

    monkeypatch.setattr("kufar_server_finder.gemini.time.sleep", sleeps.append)
    analyzer = GeminiAnalyzer(
        make_config(max_retries=3),
        client_factory=lambda key: FakeClient(key, handler),
    )

    assert analyzer.analyze_ads([{"link": "x"}]) == []
    assert calls == ["key-1", "key-1", "key-1"]
    assert sleeps == [1, 2]


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
    assert used_keys == ["key-1", "key-1", "key-1"]


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


def test_dynamic_scheduler_gives_next_task_to_free_worker():
    first_started = threading.Event()
    fourth_started = threading.Event()

    def handler(key, kwargs):
        link = parse_chunk_link(kwargs["contents"])
        if link == "link-1":
            first_started.set()
            assert fourth_started.wait(timeout=2)
        elif link == "link-4":
            assert first_started.wait(timeout=2)
            fourth_started.set()
        return analysis_json(link)

    analyzer = GeminiAnalyzer(
        make_config(worker_count=3, chunk_size=1),
        client_factory=lambda key: FakeClient(key, handler),
    )
    try:
        result = analyzer.analyze_ads(
            [{"link": f"link-{index}"} for index in range(1, 5)]
        )
    finally:
        analyzer.close()

    assert [item.link for item in result] == [
        "link-1",
        "link-2",
        "link-3",
        "link-4",
    ]


def test_image_downloads_run_in_parallel_and_keep_order():
    barrier = threading.Barrier(3)

    class ParallelImageSession:
        def __init__(self):
            self.headers = {}
            self.thread_names = []

        def get(self, url, timeout):
            self.thread_names.append(threading.current_thread().name)
            barrier.wait(timeout=2)
            return FakeImageResponse(content=url.encode())

    session = ParallelImageSession()
    analyzer = GeminiAnalyzer(
        make_config(
            worker_count=1,
            image_download_workers=3,
            vision_max_images=3,
        ),
        client=FakeClient("single", lambda key, kwargs: "[]"),
        image_session=session,
    )
    try:
        parts = analyzer._download_image_parts(
            {
                "images": [
                    "https://example/1.jpg",
                    "https://example/2.jpg",
                    "https://example/3.jpg",
                ]
            }
        )
    finally:
        analyzer.close()

    payloads = [
        getattr(part, "data", None)
        or getattr(getattr(part, "inline_data", None), "data", None)
        for part in parts
    ]
    assert payloads == [
        b"https://example/1.jpg",
        b"https://example/2.jpg",
        b"https://example/3.jpg",
    ]
    assert len(set(session.thread_names)) == 3


def test_adaptive_chunks_limit_items_and_serialized_characters():
    payload = [
        {"link": "1", "description": "a" * 20},
        {"link": "2", "description": "b" * 20},
        {"link": "3", "description": "c"},
    ]

    by_count = GeminiAnalyzer._build_adaptive_chunks(
        payload,
        max_items=2,
        max_chars=10_000,
    )
    by_chars = GeminiAnalyzer._build_adaptive_chunks(
        payload,
        max_items=10,
        max_chars=60,
    )

    assert [(start, len(chunk)) for start, chunk in by_count] == [(0, 2), (2, 1)]
    assert [(start, len(chunk)) for start, chunk in by_chars] == [
        (0, 1),
        (1, 1),
        (2, 1),
    ]


def test_executors_are_reused_and_context_manager_closes_analyzer():
    def handler(key, kwargs):
        contents = kwargs["contents"]
        link = parse_chunk_link(contents)
        if contents.startswith("Нормализуй"):
            return json.dumps(
                [{"link": link, "normalized_cpu_model": "Intel Core i5-3470"}]
            )
        return analysis_json(link)

    analyzer = GeminiAnalyzer(
        make_config(worker_count=1),
        client_factory=lambda key: FakeClient(key, handler),
    )
    executor_id = id(analyzer._task_executor)

    with analyzer as active:
        assert active is analyzer
        assert analyzer.analyze_ads([{"link": "x"}])[0].link == "x"
        assert analyzer.normalize_cpu_names(
            [{"link": "x", "cpu_model": "i5 3470"}]
        )[0].normalized_cpu_model == "Intel Core i5-3470"
        assert id(analyzer._task_executor) == executor_id
        assert analyzer.analyze_ads([]) == []

    assert analyzer._closed is True
    analyzer.close()  # повторное закрытие безопасно
    with pytest.raises(RuntimeError, match="закрыт"):
        analyzer._run_parallel(
            [1],
            operation=lambda worker, task: task,
            fallback=lambda: 0,
        )
