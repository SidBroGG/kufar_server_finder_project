from __future__ import annotations

from typing import Any

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
        description="Точная модель процессора либо платформа/сокет",
    )
    ram_type: str | None = Field(
        default=None,
        description="DDR2, DDR3, DDR4, DDR5 либо null",
    )
    ram_gb: int | None = Field(
        default=None,
        description="Объём оперативной памяти в ГБ либо null",
    )
