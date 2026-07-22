# Kufar Server Finder

Программа собирает объявления Kufar, фильтрует рабочие компьютеры через Gemini,
извлекает характеристики, анализирует фотографии, добавляет CPU Benchmark и
экспортирует результат в Excel.

## Требования

- Python 3.11+;
- Windows, Linux или macOS;
- один API-ключ Gemini.

## Установка в Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Заполните `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_WORKER_COUNT=3
```

## Настройки Gemini

Все AI worker используют один `GEMINI_API_KEY`. Для каждого worker создаётся
отдельный клиент, поэтому задачи могут выполняться параллельно.

Основные параметры:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_WORKER_COUNT=3
GEMINI_ANALYSIS_MODEL=gemini-3.1-flash-lite
GEMINI_SPECS_MODEL=gemini-3.1-flash-lite
GEMINI_VISION_MODEL=gemini-3.1-flash-lite
GEMINI_CHUNK_SIZE=30
GEMINI_SPECS_CHUNK_SIZE=25
GEMINI_REQUEST_DELAY=1
GEMINI_MAX_RETRIES=3
GEMINI_VISION_MAX_IMAGES=5
GEMINI_IMAGE_TIMEOUT=20
```

`GEMINI_WORKER_COUNT` задаёт количество параллельных AI worker. Значение должно
быть больше нуля. Большое количество worker может быстрее исчерпать лимит одного
API-ключа.

`GEMINI_REQUEST_DELAY` задаёт задержку между последовательными задачами одного
worker. `GEMINI_MAX_RETRIES` — максимальное число попыток одного запроса, включая
первую. При ошибке `429` запрос повторяется на том же ключе с увеличивающейся
задержкой.

Для собственного endpoint или прокси можно необязательно задать:

```env
GEMINI_BASE_URL=https://your-gemini-endpoint.example
GEMINI_API_VERSION=v1beta
```

Если параметры не указаны, используются стандартные настройки SDK Gemini.

## Команды

### Сбор объявлений

```powershell
python -m kufar_server_finder collect --computers-only --max-price 50 --output output_unfiltered.json
```

`--computers-only` включает категории компьютеров и ноутбуков. `--max-price`
передаётся в API Kufar и дополнительно проверяется локально.

### Анализ текста

```powershell
python -m kufar_server_finder analyze --input output_unfiltered.json --output output.json --extract-specs
```

Флаг `--infer-specs` сохранён как псевдоним `--extract-specs`.

### Анализ фотографий

```powershell
python -m kufar_server_finder vision --input output.json --output output_vision.json
```

Фото-анализ заполняет или уточняет CPU, сокет, тип и объём ОЗУ, тип устройства и
примерное максимальное потребление системы. Для приблизительных значений
сохраняются `*_confidence` и `*_source`.

### Полный pipeline без повторного сбора

```powershell
python -m kufar_server_finder pipeline --input output_unfiltered.json --output output.json --excel-output output.xlsx --extract-specs --dataset CPU_benchmark_v4.csv
```

Порядок этапов:

```text
analyze -> vision -> benchmark -> JSON -> Excel
```

Если `--dataset` не указан, этап CPU Benchmark пропускается.

### Сбор и полная обработка

```powershell
python -m kufar_server_finder run --computers-only --max-price 50 --raw-output output_unfiltered.json --output output.json --excel-output output.xlsx --extract-specs --dataset CPU_benchmark_v4.csv
```

### Только CPU Benchmark

```powershell
python -m kufar_server_finder benchmark --input output_vision.json --output output_benchmark.json --dataset CPU_benchmark_v4.csv
```

Перед локальным поиском Gemini исправляет очевидные опечатки в названии CPU, но
не должен менять номер модели или буквенный суффикс. Локальный поиск учитывает
разное написание `Core2/Core 2`, `DualCore/Dual Core`, перестановку слов и частые
опечатки.

### Только Excel

```powershell
python -m kufar_server_finder excel --input output.json --output output.xlsx
```

В Excel выводятся тип устройства, цена, ОЗУ, сокет, CPU, CPU Benchmark,
потребление системы и ссылка. Отсутствующие значения заменяются прочерком.
Уверенность `low`, `medium`, `high` отмечается красным, оранжевым и зелёным.

## Тесты

```powershell
python -m pip install -e ".[dev]"
pytest
```

Или:

```powershell
python -m pip install -r requirements-dev.txt
pytest
```

`pytest` запускает покрытие и завершается ошибкой, если покрытие пакета ниже 95%.
