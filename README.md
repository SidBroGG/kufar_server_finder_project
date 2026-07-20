# Kufar Server Finder

Небольшой проект для:

1. сбора объявлений Kufar с сортировкой по цене;
2. AI-фильтрации рабочих устройств, подходящих под Debian-сервер;
3. опционального определения CPU, типа и объёма ОЗУ.

Архитектура намеренно простая:

- `kufar.py` — работа с Kufar;
- `gemini.py` — обращения к Gemini и валидация ответов;
- `pipeline.py` — бизнес-логика фильтрации и объединения результатов;
- `config.py` — настройки;
- `cli.py` — команды запуска;
- `storage.py` — чтение и запись JSON.

## Установка

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
pip install -e .
cp .env.example .env
```

На Windows вместо `cp` можно выполнить:

```powershell
Copy-Item .env.example .env
```

В `.env` укажите новый Gemini API-ключ:

```env
GEMINI_API_KEY=...
```

## Запуск

Собрать объявления без AI:

```bash
python -m kufar_server_finder collect \
  --computers-only \
  --max-price 50 \
  --output output_unfiltered.json
```

Обработать ранее собранный JSON:

```bash
python -m kufar_server_finder analyze \
  --input output_unfiltered.json \
  --output output.json \
  --infer-specs
```

Полный цикл:

```bash
python -m kufar_server_finder run \
  --computers-only \
  --max-price 50 \
  --raw-output output_unfiltered.json \
  --output output.json \
  --infer-specs
```

После `pip install -e .` доступна и короткая команда:

```bash
kufar-server-finder run --computers-only --max-price 50
```

## Полезные параметры

- `--query "mini pc"` — поисковый запрос;
- `--no-descriptions` — ускорить сбор, не открывая страницы объявлений;
- `--page-delay 0.5` — пауза между страницами поиска;
- `--detail-delay 0.5` — пауза между страницами объявлений;
- `--timeout 20` — HTTP-таймаут;
- `--verbose` — подробные логи (параметр ставится до команды).

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

## Безопасность

API-ключ не хранится в исходном коде и файл `.env` исключён из Git.
Ключ, который был встроен в исходный скрипт, следует отозвать и выпустить заново.
