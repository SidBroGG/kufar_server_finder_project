from typing import Any

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
    assert client._parse_ad(
        {"ad_link": "https://example/zero", "price_byn": "0"},
        load_descriptions=False,
    ) is None


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


def test_expensive_ad_does_not_hide_cheaper_ads_below_it():
    result = FakeKufarClient().fetch_ads(
        max_price=50,
        load_descriptions=False,
    )
    assert [ad["price"] for ad in result] == [10.0, 20.0]

def test_search_params_include_server_side_max_price():
    client = KufarClient(KufarConfig(page_delay=0, detail_delay=0))
    params = client._build_search_params(
        None,
        "16020",
        max_price=20,
    )
    assert params["prc"] == "r:0,2000"


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


def test_expensive_ads_do_not_load_description():
    client = DescriptionTrackingClient()
    result = client.fetch_ads(max_price=20, load_descriptions=True)
    assert [ad["price"] for ad in result] == [10.0]
    assert client.description_links == ["https://example/10"]

class ExpensivePageStopClient(KufarClient):
    def __init__(self) -> None:
        super().__init__(KufarConfig(page_delay=0, detail_delay=0))
        self.calls = 0

    def _get_json(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "ads": [{"ad_link": "https://example/10", "price_byn": "1000"}],
                "pagination": {
                    "pages": [{"label": "next", "token": "page-2"}]
                },
            }
        return {
            "ads": [{"ad_link": "https://example/100", "price_byn": "10000"}],
            "pagination": {
                "pages": [{"label": "next", "token": "page-3"}]
            },
        }


def test_pagination_stops_on_first_fully_expensive_page():
    client = ExpensivePageStopClient()
    result = client.fetch_ads(max_price=20, load_descriptions=False)
    assert [ad["price"] for ad in result] == [10.0]
    assert client.calls == 2

import requests
import pytest


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

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_default_init_and_search_params_variants():
    session = FakeSession([])
    client = KufarClient(session=session)
    assert client.config.region == "7"
    assert session.headers["Accept"] == "application/json"

    params = client._build_search_params("server", "16020", max_price=None)
    assert params["query"] == "server"
    assert params["cat"] == "16020"
    assert "prc" not in params
    assert client._build_search_params(None, None, max_price=-1)["prc"] == "r:0,0"


def test_parse_ad_builds_characteristics_images_and_description_statuses(monkeypatch):
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

    monkeypatch.setattr(client, "_fetch_description", lambda link: "Описание")
    loaded = client._parse_ad(raw, load_descriptions=True)
    assert loaded["description_status"] == "loaded"
    assert loaded["description"] == "Описание"
    assert loaded["characteristics"] == {"ОЗУ": "8 ГБ"}
    assert loaded["images"] == [f"{client.IMAGE_BASE_URL}/a.jpg"]

    monkeypatch.setattr(client, "_fetch_description", lambda link: "")
    missing = client._parse_ad({**raw, "subject": ""}, load_descriptions=True)
    assert missing["title"] == "Без названия"
    assert missing["description_status"] == "missing"

    monkeypatch.setattr(client, "_fetch_description", lambda link: None)
    failed = client._parse_ad(raw, load_descriptions=True)
    assert failed["description_status"] == "load_error"
    assert failed["description_load_error"] is True

    not_requested = client._parse_ad(raw, load_descriptions=False)
    assert not_requested["description_status"] == "not_requested"
    assert client._parse_ad({"price_byn": "100"}, load_descriptions=False) is None


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

    assert client._get_json("https://api") == {"ads": []}
    with pytest.raises(ValueError, match="неожиданного формата"):
        client._get_json("https://api")
    assert client._fetch_description("https://item/1") == "A\nB"
    assert client._fetch_description("https://item/2") == ""
    assert client._fetch_description("https://item/3") is None


def test_fetch_ads_computers_categories_deduplicates_and_uses_cursor(monkeypatch):
    class MultiCategoryClient(KufarClient):
        def __init__(self):
            super().__init__(KufarConfig(page_delay=0, detail_delay=0))
            self.requests = []

        def _get_json(self, url, **kwargs):
            params = dict(kwargs["params"])
            self.requests.append(params)
            category = params.get("cat")
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

    client = MultiCategoryClient()
    result = client.fetch_ads(
        query="pc",
        computers_only=True,
        max_price=20,
        load_descriptions=False,
    )

    assert len(result) == 1
    assert {request["cat"] for request in client.requests} == {"16020", "16040"}
    assert sum("cursor" in request for request in client.requests) == 2


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
        result = client.fetch_ads(max_price=10, load_descriptions=True)
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

    assert client._fetch_description("https://example/item") is None
    assert client._descriptions_disabled.is_set()
    assert len(session.calls) == 2
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

    assert client._fetch_description("https://example/item") == "OK"
    assert not client._descriptions_disabled.is_set()
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
