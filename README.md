# Kufar Server Finder

CLI-программа для поиска недорогих рабочих компьютеров и ноутбуков на Kufar.
Она собирает объявления из двух категорий, загружает описания, фильтрует товары
через Gemini, определяет комплектующие по тексту и фотографиям, добавляет CPU
Benchmark и экспортирует результат в Excel.

## Что выполняется автоматически

- сбор категорий `16020` (компьютеры) и `16040` (ноутбуки);
- загрузка описания каждого подходящего объявления;
- фильтрация рабочих устройств;
- извлечение CPU, ОЗУ и сокета из текста;
- уточнение характеристик по фотографиям;
- заполнение примерных значений, когда точные определить невозможно;
- поиск CPU в CSV-датасете Benchmark, если передан `--dataset`;
- экспорт итогового JSON в Excel для команд `run` и `pipeline`.

Текстовый поиск отключён. Параметров `query`, `--computers-only`,
`--no-descriptions`, `--extract-specs` и `--infer-specs` больше нет.

## Требования

- Python 3.11 или новее;
- Windows 10, Linux или macOS;
- API-ключ Gemini.

## Установка в Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Откройте `.env` и укажите ключ:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## Настройки `.env`

### Kufar

```env
KUFAR_REGION=7
KUFAR_TIMEOUT=20
KUFAR_DETAIL_DELAY=1
KUFAR_DETAIL_WORKERS=3
KUFAR_DETAIL_RETRIES=3
```

- `KUFAR_REGION` — идентификатор региона Kufar;
- `KUFAR_TIMEOUT` — тайм-аут HTTP-запроса в секундах;
- `KUFAR_DETAIL_DELAY` — общая задержка между запросами страниц объявлений;
- `KUFAR_DETAIL_WORKERS` — число параллельных загрузчиков описаний;
- `KUFAR_DETAIL_RETRIES` — число попыток загрузки описания.

При частых ответах `429` увеличьте `KUFAR_DETAIL_DELAY` или уменьшите
`KUFAR_DETAIL_WORKERS`.

### Gemini

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_WORKER_COUNT=3
GEMINI_ANALYSIS_MODEL=gemini-3.5-flash-lite
GEMINI_SPECS_MODEL=gemini-3.5-flash-lite
GEMINI_VISION_MODEL=gemini-3.5-flash-lite
GEMINI_CHUNK_SIZE=30
GEMINI_SPECS_CHUNK_SIZE=25
GEMINI_MAX_CHUNK_CHARS=25000
GEMINI_SPECS_MAX_CHUNK_CHARS=20000
GEMINI_VISION_MAX_IMAGES=5
GEMINI_IMAGE_DOWNLOAD_WORKERS=3
GEMINI_IMAGE_TIMEOUT=20
GEMINI_REQUEST_DELAY=1
GEMINI_MAX_RETRIES=3
```

Для собственного endpoint или прокси доступны:

```env
GEMINI_BASE_URL=https://your-gemini-endpoint.example
GEMINI_API_VERSION=v1beta
```

## Команды

### Полный запуск

```powershell
python -m kufar_server_finder run --max-price 50 --raw-output output_unfiltered.json --output output.json --excel-output output.xlsx --dataset CPU_benchmark_v4.csv
```

Этапы:

```text
сбор категорий -> описания -> AI-фильтрация и характеристики -> фото-анализ -> benchmark -> JSON -> Excel
```

`--dataset` необязателен. Без него этап CPU Benchmark пропускается.

### Только сбор объявлений

```powershell
python -m kufar_server_finder collect --max-price 50 --output output_unfiltered.json
```

Доступные параметры сбора:

- `--max-price` — максимальная цена в BYN;
- `--page-delay` — задержка перед загрузкой следующей страницы категории.

Остальные сетевые настройки Kufar задаются только через `.env`.

### Анализ собранного JSON

```powershell
python -m kufar_server_finder analyze --input output_unfiltered.json --output output_analyzed.json --dataset CPU_benchmark_v4.csv
```

Команда всегда фильтрует объявления и извлекает характеристики из текста.

### Полный pipeline без повторного сбора

```powershell
python -m kufar_server_finder pipeline --input output_unfiltered.json --output output.json --excel-output output.xlsx --dataset CPU_benchmark_v4.csv
```

### Только анализ фотографий

```powershell
python -m kufar_server_finder vision --input output_analyzed.json --output output_vision.json
```

### Только CPU Benchmark

```powershell
python -m kufar_server_finder benchmark --input output_vision.json --output output_benchmark.json --dataset CPU_benchmark_v4.csv
```

### Только Excel

```powershell
python -m kufar_server_finder excel --input output.json --output output.xlsx
```

В Excel выводятся тип устройства, цена, ОЗУ, сокет, CPU, CPU Benchmark,
оценочная мощность системы и ссылка. Цвет ячейки показывает уверенность:
красный — `low`, оранжевый — `medium`, зелёный — `high`.

## Тесты

```powershell
python -m pip install -e ".[dev]"
pytest
```

Конфигурация `pytest` проверяет покрытие пакета и требует минимум 95%.
