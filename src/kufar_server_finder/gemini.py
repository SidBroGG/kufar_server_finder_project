from __future__ import annotations

import json
import logging
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
    key_numbers: tuple[int, ...]
    clients: tuple[Any, ...]
    image_session: requests.Session
    client_index: int = 0

    @property
    def client(self) -> Any:
        return self.clients[self.client_index]

    @property
    def current_key_number(self) -> int:
        return self.key_numbers[self.client_index]

    def rotate_api_key(self) -> None:
        self.client_index = (self.client_index + 1) % len(self.clients)


class GeminiAnalyzer:
    def __init__(
        self,
        config: GeminiConfig,
        client: genai.Client | None = None,
        image_session: requests.Session | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        factory = client_factory or (lambda api_key: genai.Client(api_key=api_key))

        if client is not None:
            # Сохраняет совместимость с тестами и внешним кодом, который передаёт
            # один заранее созданный клиент. Обычный запуск всегда использует 3 worker.
            groups = ((config.api_key,),)
            client_groups: tuple[tuple[Any, ...], ...] = ((client,),)
            key_number_groups = ((1,),)
        else:
            groups = config.worker_api_key_groups
            client_groups = tuple(
                tuple(factory(api_key) for api_key in group) for group in groups
            )
            key_number_groups = (
                (1, 2, 3),
                (4, 5, 6),
                (7, 8, 9),
            )

        self._worker_api_key_groups = groups
        sessions = self._build_image_sessions(len(groups), image_session)
        self._workers = tuple(
            _GeminiWorker(
                number=index + 1,
                key_numbers=key_number_groups[index],
                clients=client_groups[index],
                image_session=sessions[index],
            )
            for index in range(len(groups))
        )

    @property
    def worker_api_key_groups(self) -> tuple[tuple[str, ...], ...]:
        return self._worker_api_key_groups

    @property
    def client(self) -> Any:
        """Совместимость: текущий клиент первого worker."""
        return self._workers[0].client

    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]:
        payload = [self._analysis_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.chunk_size,
            model=self.config.analysis_model,
            instruction=ANALYSIS_SYSTEM_INSTRUCTION,
            response_model=AdAnalysis,
            prompt_prefix="Проанализируй объявления",
        )

    def extract_explicit_specs(
        self, ads: list[dict[str, Any]]
    ) -> list[PCComponentSpec]:
        payload = [self._specs_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.specs_chunk_size,
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
        model: str,
        instruction: str,
        response_model: type[ModelT],
        prompt_prefix: str,
    ) -> list[ModelT]:
        if not payload:
            return []

        total = len(payload)
        tasks = [
            (start, payload[start : start + chunk_size])
            for start in range(0, total, chunk_size)
        ]

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

    def _run_parallel(
        self,
        tasks: list[TaskT],
        *,
        operation: Callable[[_GeminiWorker, TaskT], ResultT],
        fallback: Callable[[], ResultT],
    ) -> list[ResultT]:
        if not tasks:
            return []

        executors = [
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"gemini-worker-{worker.number}",
            )
            for worker in self._workers
        ]
        futures: list[Future[ResultT]] = []
        submitted_per_worker = [0] * len(self._workers)

        try:
            for index, task in enumerate(tasks):
                worker_index = index % len(self._workers)
                worker = self._workers[worker_index]
                should_delay = submitted_per_worker[worker_index] > 0
                submitted_per_worker[worker_index] += 1
                futures.append(
                    executors[worker_index].submit(
                        self._execute_worker_task,
                        worker,
                        task,
                        should_delay,
                        operation,
                    )
                )

            result: list[ResultT] = []
            for future in futures:
                try:
                    result.append(future.result())
                except Exception as exc:
                    logger.exception("Необработанная ошибка Gemini worker: %s", exc)
                    result.append(fallback())
            return result
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=False)

    def _execute_worker_task(
        self,
        worker: _GeminiWorker,
        task: TaskT,
        should_delay: bool,
        operation: Callable[[_GeminiWorker, TaskT], ResultT],
    ) -> ResultT:
        if should_delay and self.config.request_delay:
            time.sleep(self.config.request_delay)
        return operation(worker, task)

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
        rate_limit_attempts = 0
        other_attempts = 0
        max_rate_limit_attempts = self.config.max_retries * len(worker.clients)

        while True:
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

                if self._is_rate_limit_error(exc):
                    rate_limit_attempts += 1
                    previous_key = worker.current_key_number
                    worker.rotate_api_key()
                    logger.warning(
                        "Worker %s: Gemini 429 на ключе %s; следующий ключ %s; "
                        "попытка %s/%s",
                        worker.number,
                        previous_key,
                        worker.current_key_number,
                        rate_limit_attempts,
                        max_rate_limit_attempts,
                    )
                    if rate_limit_attempts < max_rate_limit_attempts:
                        continue
                    break

                other_attempts += 1
                logger.warning(
                    "Worker %s: ошибка Gemini на ключе %s, попытка %s/%s: %s",
                    worker.number,
                    worker.current_key_number,
                    other_attempts,
                    self.config.max_retries,
                    exc,
                )
                if other_attempts >= self.config.max_retries:
                    break
                time.sleep(min(2 ** (other_attempts - 1), 8))

        logger.error(
            "Worker %s: запрос пропущен после повторных ошибок: %s",
            worker.number,
            last_error,
        )
        return None

    def _rotate_api_key(self, worker_index: int = 0) -> None:
        """Совместимость с прежним приватным методом."""
        self._workers[worker_index].rotate_api_key()

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
        parts: list[types.Part] = []
        image_urls = ad.get("images") or []
        session = image_session or self._workers[0].image_session

        for url in image_urls[: self.config.vision_max_images]:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
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
                    continue
                parts.append(
                    types.Part.from_bytes(
                        data=response.content,
                        mime_type=mime_type,
                    )
                )
            except requests.RequestException as exc:
                logger.warning("Не удалось загрузить фото %s: %s", url, exc)

        return parts

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
