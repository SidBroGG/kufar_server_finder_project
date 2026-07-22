from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections.abc import Iterator
from typing import Any

import requests
from bs4 import BeautifulSoup

from .config import KufarConfig

logger = logging.getLogger(__name__)
_MAX_RETRY_AFTER_SECONDS = 15.0


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
        self._owns_session = session is None
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.config.user_agent, "Accept": "application/json"}
        )
        self._detail_executor = ThreadPoolExecutor(
            max_workers=self.config.detail_workers,
            thread_name_prefix="kufar-detail",
        )
        self._closed = False
        self._interrupted = False
        self._stop_event = threading.Event()
        self._descriptions_disabled = threading.Event()
        self._detail_rate_lock = threading.Lock()
        self._next_detail_request_at = 0.0
        self._consecutive_rate_limits = 0
        self._active_detail_futures: set[Future[str | None]] = set()

    def close(self, *, wait: bool | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        for future in tuple(self._active_detail_futures):
            future.cancel()
        if wait is None:
            wait = not self._interrupted
        self._detail_executor.shutdown(wait=wait, cancel_futures=True)
        if self._owns_session:
            self.session.close()

    def abort(self) -> None:
        self._interrupted = True
        self._stop_event.set()
        for future in tuple(self._active_detail_futures):
            future.cancel()
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "KufarClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if not getattr(self, "_closed", True):
                self.close(wait=False)
        except Exception:
            pass

    def fetch_ads(
        self,
        *,
        max_price: float = 100.0,
    ) -> list[dict[str, Any]]:
        results = list(self.iter_ads(max_price=max_price))
        results.sort(key=lambda ad: ad["price"])
        logger.info("Собрано объявлений: %s", len(results))
        return results

    def iter_ads(
        self,
        *,
        max_price: float = 100.0,
    ) -> Iterator[dict[str, Any]]:
        """Отдаёт объявления по мере завершения загрузки их описаний."""
        categories = [
            self.config.category_computers,
            self.config.category_laptops,
        ]

        seen_links: set[str] = set()

        for category in categories:
            params = self._build_search_params(
                category,
                max_price=max_price,
            )
            page_number = 1
            logger.info(
                "Запуск парсера: category=%r, max_price=%s BYN",
                category,
                max_price,
            )

            while True:
                logger.info("Загрузка категории %r, страница #%s", category, page_number)
                if self.config.page_delay:
                    time.sleep(self.config.page_delay)

                data = self._get_json(self.SEARCH_URL, params=params)
                ads = data.get("ads") or []
                if not ads:
                    break

                page_has_eligible_price = False
                page_results: list[dict[str, Any]] = []
                for raw_ad in ads:
                    raw_price = self._parse_price(raw_ad.get("price_byn"))
                    if raw_price is None or raw_price > max_price:
                        continue
                    page_has_eligible_price = True

                    parsed = self._parse_ad(raw_ad)
                    if parsed is None:
                        continue

                    link = parsed["link"]
                    if link in seen_links:
                        continue
                    seen_links.add(link)
                    page_results.append(parsed)

                if page_results:
                    self._load_descriptions(page_results)
                yield from page_results

                if not page_has_eligible_price:
                    logger.info(
                        "Остановка категории %r: на странице #%s нет объявлений "
                        "до %s BYN",
                        category,
                        page_number,
                        max_price,
                    )
                    break

                cursor = self._extract_next_cursor(data)
                if not cursor:
                    break
                params["cursor"] = cursor
                page_number += 1

    def _load_descriptions(self, ads: list[dict[str, Any]]) -> None:
        if self._closed:
            raise RuntimeError("KufarClient уже закрыт")
        if self._descriptions_disabled.is_set():
            for ad in ads:
                self._apply_description_result(ad, None)
            return

        future_to_ad = {
            self._detail_executor.submit(self._fetch_description, ad["link"]): ad
            for ad in ads
        }
        self._active_detail_futures.update(future_to_ad)
        try:
            for future in as_completed(future_to_ad):
                ad = future_to_ad[future]
                try:
                    description = future.result()
                except Exception as exc:
                    logger.warning(
                        "Не удалось загрузить описание %s: %s",
                        ad.get("link"),
                        exc,
                    )
                    description = None
                self._apply_description_result(ad, description)
        except BaseException:
            self.abort()
            raise
        finally:
            self._active_detail_futures.difference_update(future_to_ad)
            # Отменённые из-за circuit breaker задания тоже должны получить статус.
            for future, ad in future_to_ad.items():
                if "description_status" not in ad or ad["description_status"] == "not_requested":
                    if future.cancelled() or self._descriptions_disabled.is_set():
                        self._apply_description_result(ad, None)

    @staticmethod
    def _apply_description_result(
        ad: dict[str, Any],
        description: str | None,
    ) -> None:
        ad.pop("description_load_error", None)
        if description is None:
            ad["description"] = ""
            ad["description_status"] = "load_error"
            ad["description_load_error"] = True
        elif description:
            ad["description"] = description
            ad["description_status"] = "loaded"
        else:
            ad["description"] = ""
            ad["description_status"] = "missing"

    def _build_search_params(
        self,
        category: str,
        *,
        max_price: float | None = None,
    ) -> dict[str, str]:
        params = {
            "rgn": self.config.region,
            "sort": "prc.a",
            "size": str(self.config.page_size),
            "lang": "ru",
            "cat": category,
        }
        if max_price is not None:
            max_price_cents = max(0, round(max_price * 100))
            params["prc"] = f"r:0,{max_price_cents}"
        return params

    def _parse_ad(
        self,
        raw_ad: dict[str, Any],
    ) -> dict[str, Any] | None:
        link = raw_ad.get("ad_link")
        if not link:
            return None

        price = self._parse_price(raw_ad.get("price_byn"))
        if price is None:
            logger.debug("Объявление пропущено из-за некорректной цены: %s", link)
            return None

        return {
            "title": raw_ad.get("subject") or "Без названия",
            "price": price,
            "link": link,
            "images": self._parse_images(raw_ad),
            "description": "",
            "description_status": "not_requested",
            "characteristics": self._parse_characteristics(raw_ad),
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

    def _fetch_description(self, link: str) -> str | None:
        last_error: Exception | None = None
        for attempt in range(1, self.config.detail_max_retries + 1):
            if not self._wait_for_detail_slot():
                return None

            try:
                response = self.session.get(link, timeout=self.config.request_timeout)
                if getattr(response, "status_code", 200) == 429:
                    retry_after = self._retry_after_seconds(response, attempt)
                    if self._register_rate_limit(retry_after):
                        return None
                    logger.warning(
                        "Kufar вернул 429 для %s; повтор через %.1f сек. (%s/%s)",
                        link,
                        retry_after,
                        attempt,
                        self.config.detail_max_retries,
                    )
                    continue

                response.raise_for_status()
                self._reset_rate_limit_counter()
                soup = BeautifulSoup(response.text, "html.parser")
                block = soup.find(attrs={"itemprop": "description"})
                return block.get_text(separator="\n", strip=True) if block else ""
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.config.detail_max_retries:
                    delay = min(2.0 ** (attempt - 1), 8.0)
                    self._defer_detail_requests(delay)
                    continue

        if last_error is not None:
            logger.warning("Не удалось загрузить описание %s: %s", link, last_error)
        return None

    def _wait_for_detail_slot(self) -> bool:
        while not self._stop_event.is_set() and not self._descriptions_disabled.is_set():
            with self._detail_rate_lock:
                now = time.monotonic()
                delay = self._next_detail_request_at - now
                if delay <= 0:
                    self._next_detail_request_at = now + self.config.detail_delay
                    return True
            if self._stop_event.wait(min(max(delay, 0.01), 0.1)):
                return False
        return False

    def _defer_detail_requests(self, delay: float) -> None:
        with self._detail_rate_lock:
            self._next_detail_request_at = max(
                self._next_detail_request_at,
                time.monotonic() + max(delay, 0.0),
            )

    def _register_rate_limit(self, retry_after: float) -> bool:
        with self._detail_rate_lock:
            self._consecutive_rate_limits += 1
            self._next_detail_request_at = max(
                self._next_detail_request_at,
                time.monotonic() + retry_after,
            )
            blocked = (
                self._consecutive_rate_limits >= self.config.rate_limit_threshold
            )
            if blocked:
                self._descriptions_disabled.set()

        if blocked:
            logger.error(
                "Kufar временно блокирует страницы объявлений (429). "
                "Загрузка остальных описаний отключена для этого запуска. "
                "Увеличьте KUFAR_DETAIL_DELAY или уменьшите "
                "KUFAR_DETAIL_WORKERS в .env."
            )
        return blocked

    def _reset_rate_limit_counter(self) -> None:
        with self._detail_rate_lock:
            self._consecutive_rate_limits = 0

    @staticmethod
    def _retry_after_seconds(response: Any, attempt: int) -> float:
        header = str(getattr(response, "headers", {}).get("Retry-After", "")).strip()
        try:
            value = float(header)
        except ValueError:
            value = 2.0 ** attempt
        return min(max(value, 1.0), _MAX_RETRY_AFTER_SECONDS)

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
    def _parse_price(value: Any) -> float | None:
        if value is None:
            return None
        try:
            price = int(value) / 100
        except (TypeError, ValueError, OverflowError):
            return None
        return price if price > 0 else None

    @staticmethod
    def _extract_next_cursor(data: dict[str, Any]) -> str | None:
        pages = (data.get("pagination") or {}).get("pages") or []
        for page in pages:
            if page.get("label") == "next":
                return page.get("token") or page.get("cursor")
        return None





