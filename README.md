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

В `.env` укажите:

```env
GEMINI_API_KEY=...
```

## Запуск

Сбор:

```powershell
python -m kufar_server_finder collect --computers-only --max-price 50 --output output_unfiltered.json
```

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
