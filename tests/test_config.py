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
    monkeypatch.setattr("kufar_server_finder.config.load_dotenv", lambda: False)
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


def test_kufar_from_env_loads_removed_cli_parameters(monkeypatch):
    monkeypatch.setenv("KUFAR_REGION", "5")
    monkeypatch.setenv("KUFAR_TIMEOUT", "12.5")
    monkeypatch.setenv("KUFAR_PAGE_DELAY", "0.75")
    monkeypatch.setenv("KUFAR_DETAIL_DELAY", "1.5")
    monkeypatch.setenv("KUFAR_DETAIL_WORKERS", "4")
    monkeypatch.setenv("KUFAR_DETAIL_RETRIES", "6")

    config = KufarConfig.from_env()

    assert config.region == "5"
    assert config.request_timeout == 12.5
    assert config.page_delay == 0.75
    assert config.detail_delay == 1.5
    assert config.detail_workers == 4
    assert config.detail_max_retries == 6
    assert config.category_computers == "16020"
    assert config.category_laptops == "16040"


def test_kufar_from_env_uses_defaults(monkeypatch):
    monkeypatch.setattr("kufar_server_finder.config.load_dotenv", lambda: False)
    for name in (
        "KUFAR_REGION",
        "KUFAR_TIMEOUT",
        "KUFAR_PAGE_DELAY",
        "KUFAR_DETAIL_DELAY",
        "KUFAR_DETAIL_WORKERS",
        "KUFAR_DETAIL_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)

    config = KufarConfig.from_env()

    assert config.region == "7"
    assert config.request_timeout == 20
    assert config.page_delay == 1
    assert config.detail_delay == 1
    assert config.detail_workers == 3
    assert config.detail_max_retries == 3


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("KUFAR_REGION", "", "не может быть пустым"),
        ("KUFAR_TIMEOUT", "0", "больше нуля"),
        ("KUFAR_PAGE_DELAY", "-1", "не может быть отрицательным"),
        ("KUFAR_DETAIL_DELAY", "-1", "не может быть отрицательным"),
        ("KUFAR_DETAIL_WORKERS", "0", "больше нуля"),
        ("KUFAR_DETAIL_RETRIES", "0", "больше нуля"),
    ],
)
def test_kufar_from_env_validates_options(monkeypatch, name, value, message):
    monkeypatch.setattr("kufar_server_finder.config.load_dotenv", lambda: False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        KufarConfig.from_env()


def test_direct_config_validates_parallel_chunk_and_kufar_options():
    with pytest.raises(ValueError, match="GEMINI_MAX_CHUNK_CHARS"):
        GeminiConfig(api_key="key", max_chunk_chars=0)
    with pytest.raises(ValueError, match="GEMINI_REQUEST_DELAY"):
        GeminiConfig(api_key="key", request_delay=-1)
    with pytest.raises(ValueError, match="GEMINI_IMAGE_TIMEOUT"):
        GeminiConfig(api_key="key", image_timeout=0)
    with pytest.raises(ValueError, match="KUFAR_REGION"):
        KufarConfig(region=" ")
    with pytest.raises(ValueError, match="KUFAR_TIMEOUT"):
        KufarConfig(request_timeout=0)
    with pytest.raises(ValueError, match="KUFAR_PAGE_DELAY"):
        KufarConfig(page_delay=-1)
    with pytest.raises(ValueError, match="KUFAR_DETAIL_WORKERS"):
        KufarConfig(detail_workers=0)
    with pytest.raises(ValueError, match="KUFAR_DETAIL_RETRIES"):
        KufarConfig(detail_max_retries=0)
    with pytest.raises(ValueError, match="KUFAR_RATE_LIMIT_THRESHOLD"):
        KufarConfig(rate_limit_threshold=0)
    with pytest.raises(ValueError, match="KUFAR_DETAIL_DELAY"):
        KufarConfig(detail_delay=-1)



