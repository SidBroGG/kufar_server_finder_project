from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class SocketGuess:
    socket: str
    confidence: Confidence


_EXPLICIT_SOCKET_RE = re.compile(
    r"\b(?:socket\s*)?("
    r"LGA\s*\d{3,4}(?:-\d)?|"
    r"AM2\+|AM3\+|AM[2345]|FM1\+|FM2\+|FM[12]|"
    r"TR4|sTRX4|sTR5|SP[356]|"
    r"rPGA\s*\d{3,4}[A-Z]?|PGA\s*\d{3,4}|BGA\s*\d{3,4}"
    r")\b",
    re.IGNORECASE,
)
_INTEL_CORE_RE = re.compile(r"\b(?:core\s+)?i[3579][\s-]?(\d{3,5})([a-z0-9]*)\b")
_RYZEN_RE = re.compile(r"\bryzen\s+[3579](?:\s+pro)?\s+(\d{4})([a-z0-9]*)\b")
_XEON_E3_RE = re.compile(r"\bxeon\s+e3[-\s]?(\d{4})(?:\s*v([1-6]))?\b")
_XEON_E5_RE = re.compile(r"\bxeon\s+e5[-\s]?(?:16|26|46)\d{2}(?:\s*v([1-4]))?\b")
_XEON_E_RE = re.compile(r"\bxeon\s+e[-\s]?(21|22|23|24)\d{2}\b")
_XEON_W_RE = re.compile(r"\bxeon\s+w[-\s]?(12|13|21|22|24|34)\d{2}\b")
_EPYC_RE = re.compile(r"\bepyc\s+(7\d{3}|8\d{3}|9\d{3})\b")
_PENTIUM_G_RE = re.compile(r"\bpentium(?:\s+gold)?\s+g(\d{3,4})\b")
_CELERON_G_RE = re.compile(r"\bceleron\s+g(\d{3,4})\b")


def infer_socket_from_cpu(cpu_model: str | None) -> SocketGuess | None:
    """Осторожно определяет сокет по названию CPU без сетевых запросов."""
    if not cpu_model:
        return None

    text = _normalize(cpu_model)
    if not text:
        return None

    explicit = _EXPLICIT_SOCKET_RE.search(text)
    if explicit:
        return SocketGuess(_canonical_socket(explicit.group(1)), "high")

    if "threadripper" in text:
        model_match = re.search(r"\b(?:19|29|39|59|79)\d{2}[a-z]*\b", text)
        if model_match:
            series = int(model_match.group(0)[:2])
            if series in {19, 29}:
                return SocketGuess("TR4", "high")
            if series in {39, 59}:
                return SocketGuess("sTRX4", "high")
            if series == 79:
                return SocketGuess("sTR5", "high")

    epyc = _EPYC_RE.search(text)
    if epyc:
        model = int(epyc.group(1))
        if 7000 <= model < 8000:
            return SocketGuess("SP3", "medium")
        if 8000 <= model < 9000:
            return SocketGuess("SP6", "medium")
        return SocketGuess("SP5", "medium")

    xeon_e3 = _XEON_E3_RE.search(text)
    if xeon_e3:
        version = int(xeon_e3.group(2) or 1)
        if version <= 2:
            return SocketGuess("LGA1155", "high")
        if version <= 4:
            return SocketGuess("LGA1150", "high")
        return SocketGuess("LGA1151", "high")

    xeon_e5 = _XEON_E5_RE.search(text)
    if xeon_e5:
        version = int(xeon_e5.group(1) or 1)
        return SocketGuess("LGA2011" if version <= 2 else "LGA2011-3", "high")

    xeon_e = _XEON_E_RE.search(text)
    if xeon_e:
        family = int(xeon_e.group(1))
        return SocketGuess(
            {21: "LGA1151", 22: "LGA1151", 23: "LGA1200", 24: "LGA1700"}[family],
            "medium",
        )

    xeon_w = _XEON_W_RE.search(text)
    if xeon_w:
        family = int(xeon_w.group(1))
        socket = {
            12: "LGA1200",
            13: "LGA1200",
            21: "LGA2066",
            22: "LGA2066",
            24: "LGA4677",
            34: "LGA4677",
        }[family]
        return SocketGuess(socket, "medium")

    if "xeon" in text:
        scalable = re.search(r"\b(?:bronze|silver|gold|platinum)\s+(\d{4})\b", text)
        if scalable:
            generation_digit = scalable.group(1)[1]
            socket = {"1": "LGA3647", "2": "LGA3647", "3": "LGA4189", "4": "LGA4677", "5": "LGA4677"}.get(generation_digit)
            if socket:
                return SocketGuess(socket, "medium")

    core_ultra = re.search(r"\bcore\s+ultra\s+[579]\s+2\d{2}[a-z]*\b", text)
    if core_ultra and not _looks_mobile(text):
        return SocketGuess("LGA1851", "medium")

    core = _INTEL_CORE_RE.search(text)
    if core:
        model = core.group(1)
        suffix = core.group(2).upper()
        hedt = _intel_hedt_socket(model, suffix)
        if hedt:
            return hedt

        generation = _intel_generation(model)
        if generation:
            mobile = _intel_mobile_socket(generation, suffix)
            if mobile:
                return mobile
            socket = {
                1: "LGA1156",
                2: "LGA1155",
                3: "LGA1155",
                4: "LGA1150",
                5: "LGA1150",
                6: "LGA1151",
                7: "LGA1151",
                8: "LGA1151",
                9: "LGA1151",
                10: "LGA1200",
                11: "LGA1200",
                12: "LGA1700",
                13: "LGA1700",
                14: "LGA1700",
            }.get(generation)
            if socket:
                return SocketGuess(socket, "high")

    ryzen = _RYZEN_RE.search(text)
    if ryzen:
        model = ryzen.group(1)
        suffix = ryzen.group(2).upper()
        if _looks_mobile(text) or any(marker in suffix for marker in ("U", "H", "HS", "HX")):
            return SocketGuess("BGA (soldered)", "low")
        series = int(model[0])
        if series in {1, 2, 3, 4, 5}:
            return SocketGuess("AM4", "high")
        if series in {7, 8, 9}:
            return SocketGuess("AM5", "high")

    if re.search(r"\b(?:amd\s+)?fx[-\s]?[4689]\d{3}\b", text):
        return SocketGuess("AM3+", "high")
    if re.search(r"\b(?:phenom|athlon)\s+ii\b", text):
        return SocketGuess("AM3", "medium")
    if re.search(r"\bathlon\s+(?:200ge|220ge|240ge|3000g|300ge)\b", text):
        return SocketGuess("AM4", "high")

    pentium = _PENTIUM_G_RE.search(text)
    if pentium:
        return _intel_g_series_socket(int(pentium.group(1)))

    celeron = _CELERON_G_RE.search(text)
    if celeron:
        return _intel_g_series_socket(int(celeron.group(1)))

    return None


def _intel_generation(model: str) -> int | None:
    if len(model) == 4:
        return int(model[0])
    if len(model) == 5:
        return int(model[:2])
    return None


def _intel_hedt_socket(model: str, suffix: str) -> SocketGuess | None:
    number = int(model)
    if number in {980, 990} and "X" in suffix:
        return SocketGuess("LGA1366", "high")
    if 3900 <= number < 5000 and "X" in suffix:
        return SocketGuess("LGA2011", "high")
    if 5800 <= number < 7000 and ("X" in suffix or number in {5820, 5930, 5960, 6800, 6850, 6900, 6950}):
        return SocketGuess("LGA2011-3", "high")
    if (7800 <= number < 8000 or 9800 <= number < 10000 or 10900 <= number < 11000) and "X" in suffix:
        return SocketGuess("LGA2066", "high")
    return None


def _intel_mobile_socket(generation: int, suffix: str) -> SocketGuess | None:
    if not suffix:
        return None
    if generation in {2, 3} and suffix in {"M", "QM", "XM"}:
        return SocketGuess("rPGA988B", "medium")
    if generation == 4 and suffix in {"M", "MQ", "MX"}:
        return SocketGuess("rPGA947", "medium")
    if any(marker in suffix for marker in ("U", "Y", "H", "HQ", "HK", "P", "G1", "G4", "G7")):
        return SocketGuess("BGA (soldered)", "medium")
    return None


def _intel_g_series_socket(model: int) -> SocketGuess | None:
    if 400 <= model < 1000:
        return SocketGuess("LGA1156", "low")
    if 1000 <= model < 3000:
        return SocketGuess("LGA1155", "medium")
    if 3000 <= model < 4000:
        return SocketGuess("LGA1150", "medium")
    if 4000 <= model < 5000:
        return SocketGuess("LGA1151", "medium")
    if 5000 <= model < 6000:
        return SocketGuess("LGA1151", "medium")
    if 6000 <= model < 7000:
        return SocketGuess("LGA1200", "medium")
    if 7000 <= model < 8000:
        return SocketGuess("LGA1700", "medium")
    return None


def _looks_mobile(text: str) -> bool:
    return any(word in text for word in (" mobile", " laptop", " notebook"))


def _normalize(value: str) -> str:
    return " ".join(
        value.casefold()
        .replace("®", " ")
        .replace("™", " ")
        .replace("(r)", " ")
        .replace("(tm)", " ")
        .split()
    )


def _canonical_socket(value: str) -> str:
    compact = re.sub(r"\s+", "", value.upper())
    replacements = {
        "STRX4": "sTRX4",
        "STR5": "sTR5",
        "RPGA988B": "rPGA988B",
        "RPGA947": "rPGA947",
    }
    return replacements.get(compact, compact)
