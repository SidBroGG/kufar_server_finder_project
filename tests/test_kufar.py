from typing import Any

import pytest
import requests

from kufar_server_finder.config import KufarConfig
from kufar_server_finder.kufar import KufarClient


def test_extract_next_cursor_supports_token_and_cursor():
    assert KufarClient._extract_next_cursor(
        {"pagination": {"pages": [{"label": "next", "token": "abc"}]}}
    ) == "abc"
    assert KufarClient._extract_next_cursor(
        {"pagination": {"pages": [{"label": "next", "cursor": "xyz"}]}}
    ) == "xyz"


def test_zero_missing_and_invalid_prices_are_skipped():
    assert KufarClient._parse_price("1234") == 12.34
    assert KufarClient._parse_price(None) is None
    assert KufarClient._parse_price("0") is None
    assert KufarClient._parse_price("bad") is None

    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    try:
        assert client._parse_ad(
            {"ad_link": "https://example/zero", "price_byn": "0"}
        ) is None
    finally:
        client.close()


class FakeKufarClient(KufarClient):
    def __init__(self) -> None:
        super().__init__(KufarConfig(page_delay=0, detail_delay=0))

    def _get_json(self, url: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ads": [
                {"ad_link": "https://example/10", "price_byn": "1000"},
                {"ad_link": "https://example/100", "price_byn": "10000"},
                {"ad_link": "https://example/20", "price_byn": "2000"},
            ],
            "pagination": {"pages": []},
        }

    def _fetch_description(self, link: str) -> str | None:
        return f"Описание {link}"


def test_expensive_ad_does_not_hide_cheaper_ads_below_it():
    client = FakeKufarClient()
    try:
        result = client.fetch_ads(max_price=50)
    finally:
        client.close()

    assert [ad["price"] for ad in result] == [10.0, 20.0]
    assert all(ad["description_status"] == "loaded" for ad in result)


def test_search_params_are_category_only_and_include_max_price():
    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    try:
        params = client._build_search_params("16020", max_price=20)
        assert params["prc"] == "r:0,2000"
        assert params["cat"] == "16020"
        assert "query" not in params

        without_price = client._build_search_params("16040", max_price=None)
        assert without_price["cat"] == "16040"
        assert "prc" not in without_price
        assert client._build_search_params("16020", max_price=-1)["prc"] == "r:0,0"
    finally:
        client.close()


class DescriptionTrackingClient(KufarClient):
    def __init__(self) -> None:
        super().__init__(KufarConfig(page_delay=0, detail_delay=0))
        self.description_links: list[str] = []

    def _get_json(self, url: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["params"]["prc"] == "r:0,2000"
        return {
            "ads": [
                {"ad_link": "https://example/10", "price_byn": "1000"},
                {"ad_link": "https://example/100", "price_byn": "10000"},
            ],
            "pagination": {"pages": []},
        }

    def _fetch_description(self, link: str) -> str | None:
        self.description_links.append(link)
        return "Описание"


def test_fetch_ads_always_loads_descriptions_only_for_eligible_ads():
    client = DescriptionTrackingClient()
    try:
        result = client.fetch_ads(max_price=20)
    finally:
        client.close()

    assert [ad["price"] for ad in result] == [10.0]
    assert client.description_links == ["https://example/10"]
    assert result[0]["description"] == "Описание"


class ExpensivePageStopClient(KufarClient):
    def __init__(self) -> None:
        super().__init__(KufarConfig(page_delay=0, detail_delay=0))
        self.requests = []

    def _get_json(self, url: str, **kwargs: Any) -> dict[str, Any]:
        params = dict(kwargs["params"])
        self.requests.append(params)
        category = params["cat"]
        if "cursor" not in params:
            return {
                "ads": [
                    {
                        "ad_link": f"https://example/{category}/10",
                        "price_byn": "1000",
                    }
                ],
                "pagination": {"pages": [{"label": "next", "token": "page-2"}]},
            }
        return {
            "ads": [
                {
                    "ad_link": f"https://example/{category}/100",
                    "price_byn": "10000",
                }
            ],
            "pagination": {"pages": [{"label": "next", "token": "page-3"}]},
        }

    def _fetch_description(self, link: str) -> str | None:
        return "Описание"


def test_pagination_stops_on_first_fully_expensive_page_for_each_category():
    client = ExpensivePageStopClient()
    try:
        result = client.fetch_ads(max_price=20)
    finally:
        client.close()

    assert [ad["price"] for ad in result] == [10.0, 10.0]
    assert len(client.requests) == 4
    assert {request["cat"] for request in client.requests} == {"16020", "16040"}


class FakeResponse:
    def __init__(
        self,
        *,
        payload=None,
        text="",
        error=None,
        status_code=200,
        headers=None,
    ):
        self._payload = payload
        self.text = text
        self.error = error
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_default_init_and_category_search_params():
    session = FakeSession([])
    client = KufarClient(session=session)
    try:
        assert client.config.region == "7"
        assert session.headers["Accept"] == "application/json"
        params = client._build_search_params("16020", max_price=None)
        assert params["cat"] == "16020"
        assert "query" not in params
    finally:
        client.close()

    assert session.closed is False  # переданная сессия принадлежит вызывающему коду


def test_parse_ad_builds_characteristics_images_and_initial_status():
    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    raw = {
        "ad_link": "https://example/item",
        "price_byn": "1234",
        "subject": "PC",
        "ad_parameters": [
            {"pl": "ОЗУ", "vl": "8 ГБ"},
            {"pl": "Регион", "vl": "Минск"},
            {"pl": "Пусто", "vl": ""},
        ],
        "images": [{"path": "a.jpg"}, {"path": ""}],
    }

    try:
        parsed = client._parse_ad(raw)
        assert parsed["description_status"] == "not_requested"
        assert parsed["description"] == ""
        assert parsed["characteristics"] == {"ОЗУ": "8 ГБ"}
        assert parsed["images"] == [f"{client.IMAGE_BASE_URL}/a.jpg"]
        assert client._parse_ad({**raw, "subject": ""})["title"] == "Без названия"
        assert client._parse_ad({"price_byn": "100"}) is None
    finally:
        client.close()


def test_apply_description_result_covers_loaded_missing_and_error():
    ad = {"description_load_error": True}
    KufarClient._apply_description_result(ad, "Описание")
    assert ad == {"description": "Описание", "description_status": "loaded"}

    KufarClient._apply_description_result(ad, "")
    assert ad == {"description": "", "description_status": "missing"}

    KufarClient._apply_description_result(ad, None)
    assert ad["description_status"] == "load_error"
    assert ad["description_load_error"] is True


def test_get_json_and_fetch_description(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(payload={"ads": []}),
            FakeResponse(payload=[]),
            FakeResponse(text='<div itemprop="description">A<br>B</div>'),
            FakeResponse(text="<html></html>"),
            FakeResponse(error=requests.RequestException("network")),
            FakeResponse(error=requests.RequestException("network")),
            FakeResponse(error=requests.RequestException("network")),
        ]
    )
    client = KufarClient(
        KufarConfig(page_delay=0, detail_delay=0, request_timeout=7),
        session=session,
    )
    monkeypatch.setattr("kufar_server_finder.kufar.time.sleep", lambda value: None)

    try:
        assert client._get_json("https://api") == {"ads": []}
        with pytest.raises(ValueError, match="неожиданного формата"):
            client._get_json("https://api")
        assert client._fetch_description("https://item/1") == "A\nB"
        assert client._fetch_description("https://item/2") == ""
        assert client._fetch_description("https://item/3") is None
    finally:
        client.close()


def test_fetch_ads_uses_both_categories_deduplicates_and_uses_cursor():
    class MultiCategoryClient(KufarClient):
        def __init__(self):
            super().__init__(KufarConfig(page_delay=0, detail_delay=0))
            self.requests = []

        def _get_json(self, url, **kwargs):
            params = dict(kwargs["params"])
            self.requests.append(params)
            category = params["cat"]
            if "cursor" not in params:
                return {
                    "ads": [
                        {
                            "ad_link": "https://example/shared",
                            "price_byn": "1000" if category == "16020" else "1200",
                        }
                    ],
                    "pagination": {"pages": [{"label": "next", "token": "next"}]},
                }
            return {"ads": [], "pagination": {"pages": []}}

        def _fetch_description(self, link):
            return "Описание"

    client = MultiCategoryClient()
    try:
        result = client.fetch_ads(max_price=20)
    finally:
        client.close()

    assert len(result) == 1
    assert {request["cat"] for request in client.requests} == {"16020", "16040"}
    assert sum("cursor" in request for request in client.requests) == 2
    assert all("query" not in request for request in client.requests)


def test_price_and_cursor_edge_cases():
    assert KufarClient._parse_price([]) is None
    assert KufarClient._extract_next_cursor({}) is None
    assert KufarClient._extract_next_cursor(
        {"pagination": {"pages": [{"label": "prev", "token": "x"}]}}
    ) is None


def test_descriptions_are_loaded_in_parallel():
    import threading

    barrier = threading.Barrier(3)

    class ParallelDescriptionClient(KufarClient):
        def __init__(self):
            super().__init__(
                KufarConfig(
                    page_delay=0,
                    detail_delay=0,
                    detail_workers=3,
                )
            )
            self.thread_names = []

        def _get_json(self, url, **kwargs):
            category = kwargs["params"]["cat"]
            if category == "16040":
                return {"ads": [], "pagination": {"pages": []}}
            return {
                "ads": [
                    {
                        "ad_link": f"https://example/{index}",
                        "price_byn": str(index * 100),
                    }
                    for index in range(1, 4)
                ],
                "pagination": {"pages": []},
            }

        def _fetch_description(self, link):
            self.thread_names.append(threading.current_thread().name)
            barrier.wait(timeout=2)
            return f"Описание {link.rsplit('/', 1)[-1]}"

    client = ParallelDescriptionClient()
    executor_id = id(client._detail_executor)
    try:
        result = client.fetch_ads(max_price=10)
    finally:
        client.close()

    assert [ad["description"] for ad in result] == [
        "Описание 1",
        "Описание 2",
        "Описание 3",
    ]
    assert all(ad["description_status"] == "loaded" for ad in result)
    assert len(set(client.thread_names)) == 3
    assert id(client._detail_executor) == executor_id
    client.close()


def test_load_descriptions_handles_worker_exception_and_disabled_state(monkeypatch):
    client = KufarClient(
        KufarConfig(page_delay=0, detail_delay=0, detail_workers=1)
    )
    ads = [{"link": "https://example/1", "description_status": "not_requested"}]
    monkeypatch.setattr(
        client,
        "_fetch_description",
        lambda link: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    try:
        client._load_descriptions(ads)
        assert ads[0]["description_status"] == "load_error"

        client._descriptions_disabled.set()
        second = [{"link": "https://example/2"}]
        client._load_descriptions(second)
        assert second[0]["description_status"] == "load_error"
    finally:
        client.close()


def test_kufar_context_manager_and_description_failure():
    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    ad = {"link": "https://example", "description": "old"}

    with client as active:
        assert active is client
        active._apply_description_result(ad, None)

    assert ad["description_status"] == "load_error"
    assert ad["description_load_error"] is True
    assert client._closed is True


def test_rate_limit_circuit_breaker_stops_description_requests(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(status_code=429),
            FakeResponse(status_code=429),
        ]
    )
    client = KufarClient(
        KufarConfig(
            page_delay=0,
            detail_delay=0,
            detail_workers=1,
            detail_max_retries=3,
            rate_limit_threshold=2,
        ),
        session=session,
    )
    monkeypatch.setattr(client, "_wait_for_detail_slot", lambda: True)

    try:
        assert client._fetch_description("https://example/item") is None
        assert client._descriptions_disabled.is_set()
        assert len(session.calls) == 2
    finally:
        client.close()


def test_rate_limit_retry_can_recover(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            FakeResponse(text='<div itemprop="description">OK</div>'),
        ]
    )
    client = KufarClient(
        KufarConfig(
            page_delay=0,
            detail_delay=0,
            detail_workers=1,
            detail_max_retries=2,
            rate_limit_threshold=3,
        ),
        session=session,
    )
    monkeypatch.setattr(client, "_wait_for_detail_slot", lambda: True)

    try:
        assert client._fetch_description("https://example/item") == "OK"
        assert not client._descriptions_disabled.is_set()
    finally:
        client.close()


def test_retry_after_and_detail_rate_helpers(monkeypatch):
    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    try:
        assert client._retry_after_seconds(
            FakeResponse(headers={"Retry-After": "2.5"}), 1
        ) == 2.5
        assert client._retry_after_seconds(FakeResponse(headers={}), 2) == 4
        assert client._retry_after_seconds(
            FakeResponse(headers={"Retry-After": "999"}), 1
        ) == 15

        values = iter([10.0, 11.0])
        monkeypatch.setattr("kufar_server_finder.kufar.time.monotonic", lambda: next(values))
        client._defer_detail_requests(2)
        assert client._next_detail_request_at == 12
        client._reset_rate_limit_counter()
        assert client._consecutive_rate_limits == 0
    finally:
        client.close()


def test_abort_makes_close_non_blocking():
    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def shutdown(self, **kwargs):
            self.calls.append(kwargs)

    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    client._detail_executor.shutdown(wait=True, cancel_futures=True)
    fake_executor = FakeExecutor()
    client._detail_executor = fake_executor

    client.abort()
    client.close()

    assert fake_executor.calls == [{"wait": False, "cancel_futures": True}]
