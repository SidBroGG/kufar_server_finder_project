import pytest

from kufar_server_finder.config import (
    GEMINI_API_KEY_ENV_NAMES,
    GeminiConfig,
)


def _set_all_keys(monkeypatch):
    for index, name in enumerate(GEMINI_API_KEY_ENV_NAMES, start=1):
        monkeypatch.setenv(name, f"key-{index}")


def test_from_env_loads_nine_keys_and_builds_fixed_worker_groups(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_CHUNK_SIZE", "7")
    monkeypatch.setenv("GEMINI_SPECS_CHUNK_SIZE", "5")
    monkeypatch.setenv("GEMINI_REQUEST_DELAY", "0")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "4")
    monkeypatch.setenv("GEMINI_VISION_MAX_IMAGES", "6")
    monkeypatch.setenv("GEMINI_IMAGE_TIMEOUT", "12.5")

    config = GeminiConfig.from_env()

    assert config.api_keys == tuple(f"key-{index}" for index in range(1, 10))
    assert config.worker_api_key_groups == (
        ("key-1", "key-2", "key-3"),
        ("key-4", "key-5", "key-6"),
        ("key-7", "key-8", "key-9"),
    )
    assert config.chunk_size == 7
    assert config.specs_chunk_size == 5
    assert config.request_delay == 0
    assert config.max_retries == 4
    assert config.vision_max_images == 6
    assert config.image_timeout == 12.5


def test_from_env_rejects_missing_key(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY_8")

    with pytest.raises(ValueError, match="GEMINI_API_KEY_8"):
        GeminiConfig.from_env()


def test_from_env_rejects_duplicate_keys(monkeypatch):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY_9", "key-1")

    with pytest.raises(ValueError, match="уникальными"):
        GeminiConfig.from_env()


def test_worker_groups_reject_invalid_direct_configuration():
    config = GeminiConfig(api_key="one")
    with pytest.raises(ValueError, match="ровно 9"):
        _ = config.worker_api_key_groups

    duplicate = GeminiConfig(
        api_key="same",
        backup_api_keys=("same",) * 8,
    )
    with pytest.raises(ValueError, match="уникальными"):
        _ = duplicate.worker_api_key_groups


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("GEMINI_CHUNK_SIZE", "0", "больше нуля"),
        ("GEMINI_IMAGE_TIMEOUT", "0", "больше нуля"),
        ("GEMINI_REQUEST_DELAY", "-1", "не может быть отрицательным"),
    ],
)
def test_from_env_validates_numeric_options(monkeypatch, name, value, message):
    _set_all_keys(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        GeminiConfig.from_env()
