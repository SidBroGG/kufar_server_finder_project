from kufar_finder_core import (
    GeminiConfig,
    GeminiEngine,
    KufarClient,
    KufarConfig,
    load_items,
    process_streaming,
    save_items,
)

from kufar_server_finder.constants import KUFAR_CATEGORIES
from kufar_server_finder.gemini import GeminiAnalyzer


def test_application_uses_external_core_types():
    assert issubclass(GeminiAnalyzer, GeminiEngine)
    assert KUFAR_CATEGORIES == ("16020", "16040")
    assert all(
        value is not None
        for value in (
            GeminiConfig,
            KufarClient,
            KufarConfig,
            load_items,
            process_streaming,
            save_items,
        )
    )
