import pytest

from kufar_server_finder.config import GeminiConfig, KufarConfig


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
    monkeypatch.setenv("GEMINI_MAX_CHUNK_CHARS", "9000")
    monkeypatch.setenv("GEMINI_SPECS_MAX_CHUNK_CHARS", "7000")
    monkeypatch.setenv("GEMINI_IMAGE_DOWNLOAD_WORKERS", "4")

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
    assert config.max_chunk_chars == 9000
    assert config.specs_max_chunk_chars == 7000
    assert config.image_download_workers == 4


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


def test_direct_config_validates_new_parallel_and_chunk_options():
    with pytest.raises(ValueError, match="GEMINI_MAX_CHUNK_CHARS"):
        GeminiConfig(api_key="key", max_chunk_chars=0)
    with pytest.raises(ValueError, match="GEMINI_REQUEST_DELAY"):
        GeminiConfig(api_key="key", request_delay=-1)
    with pytest.raises(ValueError, match="GEMINI_IMAGE_TIMEOUT"):
        GeminiConfig(api_key="key", image_timeout=0)
    with pytest.raises(ValueError, match="KUFAR_DETAIL_WORKERS"):
        KufarConfig(detail_workers=0)
    with pytest.raises(ValueError, match="KUFAR_DETAIL_MAX_RETRIES"):
        KufarConfig(detail_max_retries=0)
    with pytest.raises(ValueError, match="KUFAR_RATE_LIMIT_THRESHOLD"):
        KufarConfig(rate_limit_threshold=0)
    with pytest.raises(ValueError, match="KUFAR_DETAIL_DELAY"):
        KufarConfig(detail_delay=-1)
