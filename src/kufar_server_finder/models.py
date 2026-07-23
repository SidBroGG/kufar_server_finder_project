from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["low", "medium", "high"]
SocketTextSource = Literal["text_exact", "description_guess", "cpu_model_guess"]
ProductType = Literal[
    "desktop_pc",
    "laptop",
    "mini_pc",
    "thin_client",
    "server",
    "workstation",
    "all_in_one",
    "motherboard_bundle",
    "other",
]


class _StrictResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class AdAnalysis(_StrictResponseModel):
    link: str
    is_target: bool
    is_working: bool
    price: float = Field(default=0, ge=0)
    cpu_model: str | None = None
    ram_type: str | None = None
    ram_gb: int | None = Field(default=None, ge=0)
    cpu_socket: str | None = None
    cpu_socket_source: SocketTextSource | None = None
    cpu_socket_confidence: Confidence | None = None


class PCComponentSpec(_StrictResponseModel):
    link: str
    price: float | None = Field(default=None, ge=0)
    cpu_model: str | None = None
    ram_type: str | None = None
    ram_gb: int | None = Field(default=None, ge=0)
    cpu_socket: str | None = None
    cpu_socket_source: SocketTextSource | None = None
    cpu_socket_confidence: Confidence | None = None


class CpuNameNormalization(_StrictResponseModel):
    link: str
    normalized_cpu_model: str | None = None


class VisionComponentSpec(_StrictResponseModel):
    link: str
    cpu_model: str | None = None
    cpu_model_confidence: Confidence | None = None
    ram_type: str | None = None
    ram_type_confidence: Confidence | None = None
    ram_gb: int | None = Field(default=None, ge=0)
    ram_gb_confidence: Confidence | None = None
    cpu_socket: str | None = None
    cpu_socket_confidence: Confidence | None = None
    product_type: ProductType | None = None
    product_type_confidence: Confidence | None = None
    estimated_system_power_w: int | None = Field(default=None, ge=0)
    estimated_system_power_w_confidence: Confidence | None = None
