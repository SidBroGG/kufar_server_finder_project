import pytest

from kufar_server_finder.config import GeminiConfig


def test_from_env_loads_single_key_workers_and_optional_http_options(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "main-key")
    monkeypatch.setenv("GEMINI_WORKER_COUNT", "4")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://proxy.example")
    monkeypatch.setenv("GEMINI_API_VERSION", "v1")
    monkeypatch.setenv("GEMINI_CHUNK_SIZE", "7")
    monkeypatch.setenv("GEMINI_SPECS_CHUNK_SIZE", "5")
    monkeypatch.setenv("GEMINI_REQUEST_DELAY", "0")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "4")
    monkeypatch.setenv("GEMINI_VISION_MAX_IMAGES", "6")
    monkeypatch.setenv("GEMINI_IMAGE_TIMEOUT", "12.5")

    config = GeminiConfig.from_env()

    assert config.api_key == "main-key"
    assert config.worker_count == 4
    assert config.base_url == "https://proxy.example"
    assert config.api_version == "v1"
    assert config.chunk_size == 7
    assert config.specs_chunk_size == 5
    assert config.request_delay == 0
    assert config.max_retries == 4
    assert config.vision_max_images == 6
    assert config.image_timeout == 12.5


def test_from_env_ignores_old_backup_key_variables(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "main-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "old-backup-key")
    monkeypatch.setenv("GEMINI_API_KEY_9", "old-backup-key-9")

    config = GeminiConfig.from_env()

    assert config.api_key == "main-key"
    assert not hasattr(config, "backup_api_keys")


def test_from_env_rejects_missing_single_key(monkeypatch):
    # Изолируем тест от локального .env разработчика: иначе load_dotenv()
    # восстановит удалённый GEMINI_API_KEY и проверка станет зависеть от окружения.
    monkeypatch.setattr(
        "kufar_server_finder.config.load_dotenv",
        lambda: False,
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiConfig.from_env()


def test_optional_http_options_are_none_when_blank(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "main-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "   ")
    monkeypatch.setenv("GEMINI_API_VERSION", "")

    config = GeminiConfig.from_env()

    assert config.base_url is None
    assert config.api_version is None


def test_direct_config_rejects_empty_key_and_worker_count():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiConfig(api_key="   ")

    with pytest.raises(ValueError, match="GEMINI_WORKER_COUNT"):
        GeminiConfig(api_key="key", worker_count=0)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("GEMINI_WORKER_COUNT", "0", "больше нуля"),
        ("GEMINI_CHUNK_SIZE", "0", "больше нуля"),
        ("GEMINI_IMAGE_TIMEOUT", "0", "больше нуля"),
        ("GEMINI_REQUEST_DELAY", "-1", "не может быть отрицательным"),
    ],
)
def test_from_env_validates_numeric_options(monkeypatch, name, value, message):
    monkeypatch.setenv("GEMINI_API_KEY", "main-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        GeminiConfig.from_env()
