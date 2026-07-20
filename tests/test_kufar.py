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
