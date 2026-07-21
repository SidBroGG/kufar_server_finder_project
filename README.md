# Kufar Server Finder

Проект выполняет обработку в независимых этапах:

1. `collect` — собирает объявления;
2. `analyze` — фильтрует рабочие устройства и извлекает характеристики из текста;
3. `vision` — анализирует фотографии;
4. `pipeline` — выполняет все этапы после сбора данных;
5. `benchmark` — отдельно добавляет CPU Benchmark;
6. `excel` — отдельно экспортирует JSON в Excel.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

## API-ключи Gemini

Для AI-команд обязательны 9 уникальных ключей:

```env
GEMINI_API_KEY=worker_1_key_1
GEMINI_API_KEY_2=worker_1_key_2
GEMINI_API_KEY_3=worker_1_key_3
GEMINI_API_KEY_4=worker_2_key_1
GEMINI_API_KEY_5=worker_2_key_2
GEMINI_API_KEY_6=worker_2_key_3
GEMINI_API_KEY_7=worker_3_key_1
GEMINI_API_KEY_8=worker_3_key_2
GEMINI_API_KEY_9=worker_3_key_3
```

Все AI-задачи распределяются по кругу между тремя параллельными worker:

```text
worker 1: ключи 1 → 2 → 3 → 1
worker 2: ключи 4 → 5 → 6 → 4
worker 3: ключи 7 → 8 → 9 → 7
```

Каждый worker выполняет свои задачи последовательно, а три worker работают
одновременно. Это сохраняет порядок результатов и ускоряет анализ текста,
извлечение характеристик, нормализацию CPU и обработку фотографий.

При ошибке `429` меняется ключ только внутри группы текущего worker. Ключи другого
worker не используются. `GEMINI_MAX_RETRIES=3` задаёт количество полных проходов
по группе: при трёх ключах один запрос получает до девяти попыток при `429`.
Остальные ошибки повторяются на текущем ключе не более указанного числа раз.

## Запуск

Сбор:

```powershell
python -m kufar_server_finder collect --computers-only --max-price 50 --output output_unfiltered.json
```

`--max-price` передаётся в поисковый API Kufar и дополнительно проверяется локально.
Страницы дорогих объявлений не открываются для загрузки описания.

Фильтрация, извлечение точных CPU/ОЗУ и определение сокета:

```powershell
python -m kufar_server_finder analyze --input output_unfiltered.json --output output.json --extract-specs
```

Старый флаг `--infer-specs` оставлен как псевдоним. CPU и ОЗУ не угадываются,
но `cpu_socket` может определяться по явно указанной модели CPU, материнской
плате, чипсету или платформе из описания.

Отдельный этап анализа фотографий:

```powershell
python -m kufar_server_finder vision --input output.json --output output_vision.json
```


Полный pipeline для уже собранного `output_unfiltered.json`, без обращений к Kufar:

```powershell
python -m kufar_server_finder pipeline --input output_unfiltered.json --output output.json --excel-output output.xlsx --extract-specs --dataset CPU_benchmark_v4.csv
```

Команда выполняет `analyze → vision → benchmark → сохранение JSON → Excel`.
Если `--dataset` не передан, этап benchmark пропускается.

Отдельное добавление benchmark в готовый JSON:

```powershell
python -m kufar_server_finder benchmark --input output_vision.json --output output_benchmark.json --dataset CPU_benchmark_v4.csv
```

Отдельный экспорт JSON в Excel:

```powershell
python -m kufar_server_finder excel --input output.json --output output.xlsx
```

Фото-анализ не перезаписывает уже точные текстовые данные. Неполные значения вроде `Intel Atom` могут быть заменены точной моделью, только если маркировка на фото читается с `confidence: "high"`. Значения с фото получают поля:

- `cpu_model_source: "image_guess"`;
- `cpu_socket_source: "image_guess"`;
- `ram_type_source: "image_guess"`;
- `ram_gb_source: "image_guess"`;
- `product_type_source: "image_guess"`;
- `estimated_system_power_w_source: "image_guess"`;
- соответствующее поле `*_confidence`: `low`, `medium` или `high`.

`product_type` принимает значения `desktop_pc`, `laptop`, `mini_pc`,
`thin_client`, `server`, `workstation`, `all_in_one`, `motherboard_bundle`
или `other`.

`estimated_system_power_w` — приблизительное максимальное потребление всей
системы по фотографиям. Это не номинал блока питания и не TDP одного CPU.

Точные значения из текста помечаются `*_source: "text_exact"`.

Для сокета используются источники:

- `text_exact` — сокет прямо написан в объявлении;
- `description_guess` — определён по плате, чипсету или платформе;
- `cpu_model_guess` — определён локально по модели процессора;
- `image_guess` — определён по фотографии.

Для догадок добавляется `cpu_socket_confidence`: `low`, `medium` или `high`.
Сокет также определяется локально после распознавания CPU по тексту или фото.

## Тесты

После обновления файлов переустановите dev-зависимости, потому что параметры
`--cov` предоставляет отдельный пакет `pytest-cov`:

```powershell
python -m pip install -e ".[dev]"
pytest
```

Альтернативная установка через отдельный файл зависимостей:

```powershell
python -m pip install -r requirements-dev.txt
pytest
```

`pytest` автоматически запускает измерение покрытия и завершится ошибкой, если
покрытие пакета опустится ниже 95%.

## Экспорт в Excel

Команды `run` и `pipeline` автоматически создают Excel-файл. Для отдельного
экспорта используется команда `excel`:

```powershell
python -m kufar_server_finder excel --input output.json --output output.xlsx
```

В Excel попадают тип устройства, цена, ОЗУ, сокет, процессор, CPU Benchmark,
оценочное потребление системы и ссылка. Отсутствующие значения заменяются прочерком.
Ячейки с уверенностью `low`, `medium`, `high` окрашиваются соответственно в
красный, оранжевый и зелёный цвет. Точные текстовые значения и benchmark из
датасета также отмечаются зелёным.

## Поиск CPU Benchmark

При использовании `--dataset` перед локальным поиском Gemini исправляет очевидные
опечатки и приводит конкретное название CPU к стандартному виду, не меняя номер
модели и буквенный суффикс. Исходное значение сохраняется в `cpu_model_original`,
а исправленное помечается `cpu_model_normalization_source: "gemini"`.

Локальный поиск дополнительно учитывает слитное и раздельное написание (`Core2` /
`Core 2`, `DualCore` / `Dual Core`), перестановку слов, необязательные слова и
частые опечатки. Номер модели сопоставляется строго: например, `i5-3470` не будет
принят за `i5-3470T`.
