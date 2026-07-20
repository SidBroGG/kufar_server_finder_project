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

Фильтрация и извлечение только точных характеристик из текста:

```powershell
python -m kufar_server_finder analyze --input output_unfiltered.json --output output.json --extract-specs
```

Старый флаг `--infer-specs` оставлен как псевдоним, но теперь тоже запрещает догадки.

Отдельный этап анализа фотографий:

```powershell
python -m kufar_server_finder vision --input output.json --output output_vision.json
```

Фото-анализ не перезаписывает точные текстовые данные. Угаданные значения получают поля:

- `cpu_model_source: "image_guess"`;
- `ram_type_source: "image_guess"`;
- `ram_gb_source: "image_guess"`;
- соответствующее поле `*_confidence`: `low`, `medium` или `high`.

Точные значения из текста помечаются `*_source: "text_exact"`.

## Тесты

```powershell
pytest
```
