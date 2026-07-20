from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KufarAd(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    price: float = 0.0
    link: str
    images: list[str] = Field(default_factory=list)
    description: str = "Описание отсутствует"
    characteristics: dict[str, Any] = Field(default_factory=dict)


class AdAnalysis(BaseModel):
    link: str = Field(description="Ссылка объявления")
    is_target: bool = Field(description="Подходит ли товар для Debian-сервера")
    is_working: bool = Field(description="Работает ли устройство")
    real_price: float = Field(description="Фактическая цена комплекта в BYN")


class PCComponentSpec(BaseModel):
    link: str = Field(description="Ссылка объявления")
    cpu_model: str | None = Field(
        default=None,
        description="Только явно указанная точная модель процессора",
    )
    ram_type: str | None = Field(
        default=None,
        description="Только явно указанный DDR2/DDR3/DDR4/DDR5 либо null",
    )
    ram_gb: int | None = Field(
        default=None,
        description="Только явно указанный объём ОЗУ в ГБ либо null",
    )


Confidence = Literal["low", "medium", "high"]


class VisionComponentSpec(BaseModel):
    link: str = Field(description="Ссылка объявления")
    cpu_model: str | None = None
    cpu_model_confidence: Confidence | None = None
    ram_type: str | None = None
    ram_type_confidence: Confidence | None = None
    ram_gb: int | None = None
    ram_gb_confidence: Confidence | None = None
