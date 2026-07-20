from kufar_server_finder.kufar import KufarClient


def test_extract_next_cursor_supports_token_and_cursor():
    assert KufarClient._extract_next_cursor(
        {"pagination": {"pages": [{"label": "next", "token": "abc"}]}}
    ) == "abc"
    assert KufarClient._extract_next_cursor(
        {"pagination": {"pages": [{"label": "next", "cursor": "xyz"}]}}
    ) == "xyz"


def test_price_is_converted_from_kopecks():
    assert KufarClient._parse_price("1234") == 12.34
    assert KufarClient._parse_price(None) == 0.0
