from __future__ import annotations

import json
import logging
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, TypeAdapter

from .config import GeminiConfig
from .models import (
    AdAnalysis,
    CpuNameNormalization,
    PCComponentSpec,
    VisionComponentSpec,
)
from .prompts import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    CPU_NAME_NORMALIZATION_SYSTEM_INSTRUCTION,
    SPECS_SYSTEM_INSTRUCTION,
    VISION_SPECS_SYSTEM_INSTRUCTION,
)
from .visual_refinement import fields_needing_visual_analysis

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)
TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}


@dataclass(slots=True)
class _GeminiWorker:
    number: int
    client: Any
    image_session: requests.Session
    has_executed_task: bool = False


class GeminiAnalyzer:
    def __init__(
        self,
        config: GeminiConfig,
        client: genai.Client | None = None,
        image_session: requests.Session | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config

        if client is not None:
            clients = (client,)
        else:
            factory = client_factory or self._create_client
            clients = tuple(
                factory(config.api_key) for _ in range(config.worker_count)
            )

        sessions = self._build_image_sessions(len(clients), image_session)
        self._workers = tuple(
            _GeminiWorker(
                number=index + 1,
                client=worker_client,
                image_session=sessions[index],
            )
            for index, worker_client in enumerate(clients)
        )
        self._available_workers: queue.Queue[_GeminiWorker] = queue.Queue()
        for worker in self._workers:
            self._available_workers.put(worker)

        # Пулы создаются один раз и переиспользуются всеми этапами pipeline.
        self._task_executor = ThreadPoolExecutor(
            max_workers=len(self._workers),
            thread_name_prefix="gemini-worker",
        )
        self._image_executor = ThreadPoolExecutor(
            max_workers=max(
                1,
                len(self._workers) * self.config.image_download_workers,
            ),
            thread_name_prefix="gemini-image",
        )
        self._closed = False

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def worker_api_keys(self) -> tuple[str, ...]:
        return tuple(self.config.api_key for _ in self._workers)

    @property
    def client(self) -> Any:
        """Совместимость: клиент первого worker."""
        return self._workers[0].client

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._task_executor.shutdown(wait=True, cancel_futures=False)
        self._image_executor.shutdown(wait=True, cancel_futures=False)

        closed_sessions: set[int] = set()
        for worker in self._workers:
            session_id = id(worker.image_session)
            if session_id in closed_sessions:
                continue
            closed_sessions.add(session_id)
            close = getattr(worker.image_session, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "GeminiAnalyzer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if not getattr(self, "_closed", True):
                self._task_executor.shutdown(wait=False, cancel_futures=True)
                self._image_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def _create_client(self, api_key: str) -> genai.Client:
        http_options_values: dict[str, str] = {}
        if self.config.base_url:
            http_options_values["base_url"] = self.config.base_url
        if self.config.api_version:
            http_options_values["api_version"] = self.config.api_version

        if not http_options_values:
            return genai.Client(api_key=api_key)
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(**http_options_values),
        )

    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]:
        payload = [self._analysis_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.chunk_size,
            max_chunk_chars=self.config.max_chunk_chars,
            model=self.config.analysis_model,
            instruction=ANALYSIS_SYSTEM_INSTRUCTION,
            response_model=AdAnalysis,
            prompt_prefix="Проанализируй объявления и извлеки характеристики",
        )

    def extract_explicit_specs(
        self, ads: list[dict[str, Any]]
    ) -> list[PCComponentSpec]:
        """Совместимость для отдельного вызова; основной pipeline использует analyze_ads."""
        payload = [self._specs_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.specs_chunk_size,
            max_chunk_chars=self.config.specs_max_chunk_chars,
            model=self.config.specs_model,
            instruction=SPECS_SYSTEM_INSTRUCTION,
            response_model=PCComponentSpec,
            prompt_prefix="Извлеки только явно написанные характеристики",
        )

    def infer_specs(self, ads: list[dict[str, Any]]) -> list[PCComponentSpec]:
        """Обратная совместимость: теперь метод не делает предположений."""
        return self.extract_explicit_specs(ads)

    def normalize_cpu_names(
        self,
        ads: list[dict[str, Any]],
    ) -> list[CpuNameNormalization]:
        payload = [
            {
                "link": ad.get("link"),
                "cpu_model": ad.get("cpu_model"),
            }
            for ad in ads
        ]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.specs_chunk_size,
            max_chunk_chars=self.config.specs_max_chunk_chars,
            model=self.config.specs_model,
            instruction=CPU_NAME_NORMALIZATION_SYSTEM_INSTRUCTION,
            response_model=CpuNameNormalization,
            prompt_prefix="Нормализуй названия процессоров перед поиском benchmark",
        )

    def infer_specs_from_images(
        self, ads: list[dict[str, Any]]
    ) -> list[VisionComponentSpec]:
        candidates = [ad for ad in ads if fields_needing_visual_analysis(ad)]
        total = len(candidates)
        tasks = list(enumerate(candidates, start=1))

        def process(
            worker: _GeminiWorker,
            task: tuple[int, dict[str, Any]],
        ) -> VisionComponentSpec | None:
            index, ad = task
            requested_fields = fields_needing_visual_analysis(ad)
            image_parts = self._download_image_parts(
                ad,
                image_session=worker.image_session,
            )
            if not image_parts:
                logger.info(
                    "Фото-анализ пропущен, изображения недоступны: %s",
                    ad.get("link"),
                )
                return None

            logger.info(
                "Worker %s: фото-анализ объявления %s из %s",
                worker.number,
                index,
                total,
            )
            prompt = json.dumps(
                {
                    "link": ad.get("link"),
                    "fields_to_analyze": requested_fields,
                    "existing_values": {
                        field: ad.get(field) for field in requested_fields
                    },
                },
                ensure_ascii=False,
            )
            contents: list[Any] = [
                "Проанализируй фотографии. Входные данные: " + prompt,
                *image_parts,
            ]
            return self._generate_single_structured(
                worker=worker,
                model=self.config.vision_model,
                contents=contents,
                instruction=VISION_SPECS_SYSTEM_INSTRUCTION,
                response_model=VisionComponentSpec,
            )

        values = self._run_parallel(
            tasks,
            operation=process,
            fallback=lambda: None,
        )
        return [value for value in values if value is not None]

    def _process_chunks(
        self,
        *,
        payload: list[dict[str, Any]],
        chunk_size: int,
        max_chunk_chars: int,
        model: str,
        instruction: str,
        response_model: type[ModelT],
        prompt_prefix: str,
    ) -> list[ModelT]:
        if not payload:
            return []

        total = len(payload)
        tasks = self._build_adaptive_chunks(
            payload,
            max_items=chunk_size,
            max_chars=max_chunk_chars,
        )

        def process(
            worker: _GeminiWorker,
            task: tuple[int, list[dict[str, Any]]],
        ) -> list[ModelT]:
            start, chunk = task
            logger.info(
                "Worker %s: AI-обработка объявлений %s–%s из %s",
                worker.number,
                start + 1,
                start + len(chunk),
                total,
            )
            prompt = f"{prompt_prefix}:\n{json.dumps(chunk, ensure_ascii=False)}"
            return self._generate_structured_list(
                worker=worker,
                model=model,
                contents=prompt,
                instruction=instruction,
                response_model=response_model,
            )

        result: list[ModelT] = []
        for chunk_result in self._run_parallel(
            tasks,
            operation=process,
            fallback=list,
        ):
            result.extend(chunk_result)
        return result

    @staticmethod
    def _build_adaptive_chunks(
        payload: list[dict[str, Any]],
        *,
        max_items: int,
        max_chars: int,
    ) -> list[tuple[int, list[dict[str, Any]]]]:
        chunks: list[tuple[int, list[dict[str, Any]]]] = []
        current: list[dict[str, Any]] = []
        current_chars = 2  # []
        current_start = 0

        for index, item in enumerate(payload):
            item_chars = len(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            separator_chars = 1 if current else 0
            exceeds_count = len(current) >= max_items
            exceeds_chars = current and (
                current_chars + separator_chars + item_chars > max_chars
            )
            if exceeds_count or exceeds_chars:
                chunks.append((current_start, current))
                current = []
                current_chars = 2
                current_start = index
                separator_chars = 0

            current.append(item)
            current_chars += separator_chars + item_chars

        if current:
            chunks.append((current_start, current))
        return chunks

    def _run_parallel(
        self,
        tasks: list[TaskT],
        *,
        operation: Callable[[_GeminiWorker, TaskT], ResultT],
        fallback: Callable[[], ResultT],
    ) -> list[ResultT]:
        if not tasks:
            return []
        if self._closed:
            raise RuntimeError("GeminiAnalyzer уже закрыт")

        futures: list[Future[ResultT]] = [
            self._task_executor.submit(
                self._execute_queued_task,
                task,
                operation,
            )
            for task in tasks
        ]

        result: list[ResultT] = []
        for future in futures:
            try:
                result.append(future.result())
            except Exception as exc:
                logger.exception("Необработанная ошибка Gemini worker: %s", exc)
                result.append(fallback())
        return result

    def _execute_queued_task(
        self,
        task: TaskT,
        operation: Callable[[_GeminiWorker, TaskT], ResultT],
    ) -> ResultT:
        worker = self._available_workers.get()
        try:
            if worker.has_executed_task and self.config.request_delay:
                time.sleep(self.config.request_delay)
            worker.has_executed_task = True
            return operation(worker, task)
        finally:
            self._available_workers.put(worker)

    def _generate_structured_list(
        self,
        *,
        worker: _GeminiWorker,
        model: str,
        contents: Any,
        instruction: str,
        response_model: type[ModelT],
    ) -> list[ModelT]:
        adapter = TypeAdapter(list[response_model])
        value = self._request_json(
            worker=worker,
            model=model,
            contents=contents,
            instruction=instruction,
            response_schema=list[response_model],
            adapter=adapter,
        )
        return value if isinstance(value, list) else []

    def _generate_single_structured(
        self,
        *,
        worker: _GeminiWorker,
        model: str,
        contents: Any,
        instruction: str,
        response_model: type[ModelT],
    ) -> ModelT | None:
        adapter = TypeAdapter(response_model)
        value = self._request_json(
            worker=worker,
            model=model,
            contents=contents,
            instruction=instruction,
            response_schema=response_model,
            adapter=adapter,
        )
        return value if isinstance(value, response_model) else None

    def _request_json(
        self,
        *,
        worker: _GeminiWorker,
        model: str,
        contents: Any,
        instruction: str,
        response_schema: Any,
        adapter: TypeAdapter[Any],
    ) -> Any | None:
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = worker.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini вернул пустой ответ")
                return adapter.validate_python(json.loads(response.text))
            except Exception as exc:  # SDK выбрасывает несколько типов ошибок
                last_error = exc
                error_name = (
                    "Gemini 429"
                    if self._is_rate_limit_error(exc)
                    else "ошибка Gemini"
                )
                logger.warning(
                    "Worker %s: %s, попытка %s/%s: %s",
                    worker.number,
                    error_name,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        logger.error(
            "Worker %s: запрос пропущен после повторных ошибок: %s",
            worker.number,
            last_error,
        )
        return None

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        for attribute in ("code", "status_code"):
            value = getattr(exc, attribute, None)
            if callable(value):
                value = value()
            try:
                if int(value) == 429:
                    return True
            except (TypeError, ValueError):
                pass

        message = str(exc).upper()
        return "429" in message or "RESOURCE_EXHAUSTED" in message

    def _download_image_parts(
        self,
        ad: dict[str, Any],
        *,
        image_session: requests.Session | None = None,
    ) -> list[types.Part]:
        session = image_session or self._workers[0].image_session
        image_urls = [
            url
            for url in (ad.get("images") or [])[: self.config.vision_max_images]
            if isinstance(url, str) and url.startswith(("http://", "https://"))
        ]
        futures = [
            self._image_executor.submit(
                self._download_single_image_part,
                session,
                url,
            )
            for url in image_urls
        ]

        parts: list[types.Part] = []
        for future in futures:
            part = future.result()
            if part is not None:
                parts.append(part)
        return parts

    def _download_single_image_part(
        self,
        session: requests.Session,
        url: str,
    ) -> types.Part | None:
        try:
            response = session.get(
                url,
                timeout=self.config.image_timeout,
            )
            response.raise_for_status()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            mime_type = mime_type.strip().lower()
            if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                logger.warning(
                    "Неподдерживаемый формат изображения %s: %s",
                    url,
                    mime_type,
                )
                return None
            return types.Part.from_bytes(
                data=response.content,
                mime_type=mime_type,
            )
        except requests.RequestException as exc:
            logger.warning("Не удалось загрузить фото %s: %s", url, exc)
            return None

    def _analysis_payload(self, ad: dict[str, Any]) -> dict[str, Any]:
        return {
            "link": ad.get("link"),
            "title": ad.get("title", ""),
            "price": ad.get("price", 0),
            "description": self._trim_description(ad),
            "characteristics": ad.get("characteristics") or {},
        }

    def _specs_payload(self, ad: dict[str, Any]) -> dict[str, Any]:
        return {
            "link": ad.get("link"),
            "title": ad.get("title", ""),
            "description": self._trim_description(ad, limit=600),
            "characteristics": ad.get("characteristics") or {},
        }

    def _trim_description(self, ad: dict[str, Any], limit: int | None = None) -> str:
        description = str(ad.get("description") or "")
        return description[: limit or self.config.max_description_chars]

    @staticmethod
    def _build_image_sessions(
        count: int,
        provided_session: requests.Session | None,
    ) -> tuple[requests.Session, ...]:
        if provided_session is not None:
            sessions = tuple(provided_session for _ in range(count))
        else:
            sessions = tuple(requests.Session() for _ in range(count))

        for session in sessions:
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124 Safari/537.36"
                    )
                }
            )
        return sessions
