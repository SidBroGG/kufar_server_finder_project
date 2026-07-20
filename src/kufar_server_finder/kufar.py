from __future__ import annotations

import logging
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from .config import KufarConfig

logger = logging.getLogger(__name__)


class KufarClient:
    SEARCH_URL = "https://api.kufar.by/search-api/v2/search/rendered-paginated"
    IMAGE_BASE_URL = "https://rms.kufar.by/v1/gallery"

    EXCLUDED_CHARACTERISTICS = {
        "Подкатегория",
        "Регион",
        "Город / Район",
        "Тип оплаты",
        "Состояние",
        "Возможен обмен",
        "Товары с Куфар Доставкой",
        "Товары с Куфар Оплатой",
    }

    def __init__(
        self,
        config: KufarConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or KufarConfig()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
            }
        )

    def fetch_ads(
        self,
        *,
        query: str | None = None,
        computers_only: bool = False,
        max_price: float = 100.0,
        load_descriptions: bool = True,
    ) -> list[dict[str, Any]]:
        params = self._build_search_params(query, computers_only)
        results: list[dict[str, Any]] = []
        page_number = 1

        logger.info(
            "Запуск парсера: query=%r, computers_only=%s, max_price=%s BYN",
            query,
            computers_only,
            max_price,
        )

        while True:
            logger.info("Загрузка страницы поиска #%s", page_number)
            if self.config.page_delay:
                time.sleep(self.config.page_delay)

            data = self._get_json(self.SEARCH_URL, params=params)
            ads = data.get("ads") or []
            if not ads:
                logger.info("Объявления закончились или не найдены")
                break

            should_stop = False
            for raw_ad in ads:
                parsed = self._parse_ad(raw_ad, load_descriptions=load_descriptions)
                if parsed is None:
                    continue

                if parsed["price"] > max_price:
                    logger.info(
                        "Остановка: %r стоит %.2f BYN и превышает лимит",
                        parsed["title"],
                        parsed["price"],
                    )
                    should_stop = True
                    break

                results.append(parsed)

            if should_stop:
                break

            cursor = self._extract_next_cursor(data)
            if not cursor:
                logger.info("Следующая страница не найдена")
                break

            params["cursor"] = cursor
            page_number += 1

        logger.info("Собрано объявлений: %s", len(results))
        return results

    def _build_search_params(
        self, query: str | None, computers_only: bool
    ) -> dict[str, str]:
        params = {
            "rgn": self.config.region,
            "sort": "prc.a",
            "size": str(self.config.page_size),
            "lang": "ru",
        }
        if query:
            params["query"] = query
        if computers_only:
            params["cat"] = self.config.category_computers
        return params

    def _parse_ad(
        self, raw_ad: dict[str, Any], *, load_descriptions: bool
    ) -> dict[str, Any] | None:
        link = raw_ad.get("ad_link")
        if not link:
            return None

        title = raw_ad.get("subject") or "Без названия"
        price = self._parse_price(raw_ad.get("price_byn"))
        characteristics = self._parse_characteristics(raw_ad)
        images = self._parse_images(raw_ad)
        description = (
            self._fetch_description(link)
            if load_descriptions
            else "Описание не загружалось"
        )

        return {
            "title": title,
            "price": price,
            "link": link,
            "images": images,
            "description": description,
            "characteristics": characteristics,
        }

    def _get_json(self, url: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.get(
            url,
            timeout=self.config.request_timeout,
            **kwargs,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Kufar вернул JSON неожиданного формата")
        return payload

    def _fetch_description(self, link: str) -> str:
        if self.config.detail_delay:
            time.sleep(self.config.detail_delay)

        try:
            response = self.session.get(link, timeout=self.config.request_timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            block = soup.find(attrs={"itemprop": "description"})
            if block:
                return block.get_text(separator="\n", strip=True)
            return "Описание отсутствует"
        except requests.RequestException as exc:
            logger.warning("Не удалось загрузить описание %s: %s", link, exc)
            return "Описание не удалось загрузить"

    @classmethod
    def _parse_characteristics(cls, raw_ad: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for parameter in raw_ad.get("ad_parameters") or []:
            name = parameter.get("pl") or ""
            value = parameter.get("vl") or ""
            if name and value and name not in cls.EXCLUDED_CHARACTERISTICS:
                result[str(name)] = str(value)
        return result

    @classmethod
    def _parse_images(cls, raw_ad: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for image in raw_ad.get("images") or []:
            path = image.get("path")
            if path:
                result.append(f"{cls.IMAGE_BASE_URL}/{path}")
        return result

    @staticmethod
    def _parse_price(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return int(value) / 100
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_next_cursor(data: dict[str, Any]) -> str | None:
        pages = (data.get("pagination") or {}).get("pages") or []
        for page in pages:
            if page.get("label") == "next":
                return page.get("token") or page.get("cursor")
        return None


