# Kufar Server Finder

CLI-приложение для поиска недорогих рабочих компьютеров и ноутбуков на Kufar.

## Требования

- Python 3.11+;
- опубликованный пакет `kufar-finder-core>=1.1,<2`;
- Gemini API key.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Укажите в `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## Полный запуск

```powershell
python -m kufar_server_finder run --min-price 10 --max-price 50 --raw-output output_unfiltered.json --output output.json --excel-output output.xlsx --dataset CPU_benchmark_v4.csv
```

`--dataset` необязателен. Диапазон цены включительный. По умолчанию используется
`0..100 BYN`.

Pipeline:

```text
KufarClient.iter_ads
  -> kufar_finder_core.process_streaming
  -> AI-фильтрация и характеристики
  -> vision
  -> CPU benchmark
  -> JSON и Excel
```

## Команды

```powershell
python -m kufar_server_finder collect --min-price 10 --max-price 50 --output raw.json
python -m kufar_server_finder analyze --input raw.json --output analyzed.json
python -m kufar_server_finder vision --input analyzed.json --output vision.json
python -m kufar_server_finder benchmark --input vision.json --output benchmark.json --dataset CPU_benchmark_v4.csv
python -m kufar_server_finder pipeline --input raw.json --output output.json --excel-output output.xlsx --dataset CPU_benchmark_v4.csv
python -m kufar_server_finder excel --input output.json --output output.xlsx
```

## Тесты

```powershell
pytest
```
