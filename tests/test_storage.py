import json

import pytest

from kufar_server_finder.storage import load_ads


def test_load_ads_rejects_non_object_item(tmp_path):
    path = tmp_path / "ads.json"
    path.write_text(json.dumps([{"link": "ok"}, "broken"]), encoding="utf-8")
    with pytest.raises(ValueError, match="индексом 1"):
        load_ads(path)
