# Kufar Server Finder

Проект выполняет обработку в независимых этапах:

1. `collect` — собирает объявления;
2. `analyze` — фильтрует рабочие устройства и при необходимости извлекает только явно написанные характеристики;
3. `vision` — отдельно анализирует фотографии и заполняет только отсутствующие характеристики.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

## API-ключи Gemini

Основной ключ обязателен. Ещё два ключа подключаются опционально:

```env
GEMINI_API_KEY=primary_key
GEMINI_API_KEY_2=backup_key_1
GEMINI_API_KEY_3=backup_key_2
```

Пустые и повторяющиеся ключи игнорируются.

При ошибке `429` запрос повторяется со следующим ключом по кругу:

```text
ключ 1 → ключ 2 → ключ 3 → ключ 1 → ...
```

`GEMINI_MAX_RETRIES=3` задаёт количество полных проходов по доступным ключам. При трёх ключах возможно до девяти попыток одного запроса. Остальные ошибки повторяются на текущем ключе не более трёх раз.

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

Фото-анализ не перезаписывает точные текстовые данные. Угаданные значения получают поля:

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

```powershell
pytest
```

## Экспорт в Excel

Команда `run` автоматически создаёт Excel-файл из итогового JSON:

```powershell
python -m kufar_server_finder run --computers-only --max-price 20 --output output.json --excel-output output.xlsx --extract-specs --dataset CPU_benchmark_v4.csv
```

В Excel попадают тип устройства, цена, ОЗУ, сокет, процессор, CPU Benchmark,
оценочное потребление системы и ссылка. Отсутствующие значения заменяются прочерком.
Ячейки с уверенностью `low`, `medium`, `high` окрашиваются соответственно в
красный, оранжевый и зелёный цвет. Точные текстовые значения и benchmark из
датасета также отмечаются зелёным.
