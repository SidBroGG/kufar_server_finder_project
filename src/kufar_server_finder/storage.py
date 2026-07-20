from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ads(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{source}: ожидается JSON-массив")
    return [item for item in data if isinstance(item, dict)]


def save_ads(path: str | Path, ads: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(ads, file, ensure_ascii=False, indent=2)
