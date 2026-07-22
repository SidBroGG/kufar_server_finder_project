from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .benchmark import CpuBenchmarkDataset
from .config import GeminiConfig, KufarConfig
from .excel_export import export_ads_json_to_excel
from .kufar import KufarClient
from .pipeline import AdPipeline
from .storage import load_ads, save_ads

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kufar-server-finder",
        description="Сбор и AI-фильтрация объявлений Kufar",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробные логи")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Только собрать объявления")
    _add_collect_arguments(collect)
    collect.add_argument("--output", default="output_unfiltered.json")

    analyze = subparsers.add_parser("analyze", help="Отфильтровать готовый JSON")
    analyze.add_argument("--input", default="output_unfiltered.json")
    analyze.add_argument("--output", default="output.json")
    _add_dataset_argument(analyze)

    vision = subparsers.add_parser(
        "vision",
        help="Отдельно уточнить характеристики по фотографиям",
    )
    vision.add_argument("--input", default="output.json")
    vision.add_argument("--output", default="output_vision.json")

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Полностью обработать уже собранный JSON без запросов к Kufar",
    )
    pipeline.add_argument("--input", default="output_unfiltered.json")
    pipeline.add_argument("--output", default="output.json")
    pipeline.add_argument(
        "--excel-output",
        default="output.xlsx",
        help="Excel-файл, создаваемый из итогового JSON",
    )
    _add_dataset_argument(pipeline)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Добавить CPU Benchmark в готовый JSON",
    )
    benchmark.add_argument("--input", default="output_vision.json")
    benchmark.add_argument("--output", default="output_benchmark.json")
    _add_dataset_argument(benchmark, required=True)

    excel = subparsers.add_parser(
        "excel",
        help="Экспортировать готовый JSON в Excel",
    )
    excel.add_argument("--input", default="output.json")
    excel.add_argument("--output", default="output.xlsx")

    run = subparsers.add_parser("run", help="Собрать объявления и сразу обработать")
    _add_collect_arguments(run)
    run.add_argument("--raw-output", default="output_unfiltered.json")
    run.add_argument("--output", default="output.json")
    run.add_argument(
        "--excel-output",
        default="output.xlsx",
        help="Excel-файл, создаваемый из итогового JSON",
    )
    _add_dataset_argument(run)

    return parser


def _add_dataset_argument(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
) -> None:
    parser.add_argument(
        "--dataset",
        default=None,
        required=required,
        help="CSV-датасет CPU Benchmark для добавления cpu_mark",
    )


def _add_collect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-price", type=float, default=100.0)
    parser.add_argument("--page-delay", type=float, default=1.0)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.command == "collect":
            ads = _collect(args)
            save_ads(args.output, ads)
            logger.info("Результат сохранён: %s", args.output)
            return 0

        if args.command == "analyze":
            ads = load_ads(args.input)
            pipeline = _build_pipeline()
            try:
                result = _analyze(ads, pipeline=pipeline)
                result = _apply_benchmark(
                    result,
                    args.dataset,
                    pipeline=pipeline,
                )
            finally:
                _close_pipeline(pipeline)
            save_ads(args.output, result)
            logger.info("Результат сохранён: %s", args.output)
            return 0

        if args.command == "vision":
            ads = load_ads(args.input)
            pipeline = _build_pipeline()
            try:
                result = _vision(ads, pipeline=pipeline)
            finally:
                _close_pipeline(pipeline)
            save_ads(args.output, result)
            logger.info("Результат фото-анализа сохранён: %s", args.output)
            return 0

        if args.command == "pipeline":
            ads = load_ads(args.input)
            pipeline = _build_pipeline()
            try:
                result = _analyze(ads, pipeline=pipeline)
                result = _vision(result, pipeline=pipeline)
                result = _apply_benchmark(
                    result,
                    args.dataset,
                    pipeline=pipeline,
                )
            finally:
                _close_pipeline(pipeline)
            save_ads(args.output, result)
            export_ads_json_to_excel(args.output, args.excel_output)
            logger.info(
                "Pipeline завершён: итог %s; Excel %s",
                args.output,
                args.excel_output,
            )
            return 0

        if args.command == "benchmark":
            ads = load_ads(args.input)
            result = _apply_benchmark(ads, args.dataset)
            save_ads(args.output, result)
            logger.info("JSON с benchmark сохранён: %s", args.output)
            return 0

        if args.command == "excel":
            export_ads_json_to_excel(args.input, args.output)
            logger.info("Excel сохранён: %s", args.output)
            return 0

        if args.command == "run":
            ads = _collect(args)
            save_ads(args.raw_output, ads)
            pipeline = _build_pipeline()
            try:
                result = _analyze(ads, pipeline=pipeline)
                result = _vision(result, pipeline=pipeline)
                result = _apply_benchmark(
                    result,
                    args.dataset,
                    pipeline=pipeline,
                )
            finally:
                _close_pipeline(pipeline)
            save_ads(args.output, result)
            export_ads_json_to_excel(args.output, args.excel_output)
            logger.info(
                "Сырые данные: %s; итог: %s; Excel: %s",
                args.raw_output,
                args.output,
                args.excel_output,
            )
            return 0
    except KeyboardInterrupt:
        logger.warning("Операция прервана пользователем")
        return 130
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Неожиданная ошибка")
        return 1

    return 1


def _collect(args: argparse.Namespace) -> list[dict]:
    config = replace(
        KufarConfig.from_env(),
        page_delay=max(args.page_delay, 0),
    )
    client = KufarClient(config)
    try:
        return client.fetch_ads(max_price=args.max_price)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _build_pipeline() -> AdPipeline:
    config = GeminiConfig.from_env()
    from .gemini import GeminiAnalyzer

    return AdPipeline(GeminiAnalyzer(config))


def _close_pipeline(pipeline: Any) -> None:
    close = getattr(pipeline, "close", None)
    if callable(close):
        close()


def _analyze(
    ads: list[dict],
    *,
    pipeline: AdPipeline | None = None,
) -> list[dict]:
    owns_pipeline = pipeline is None
    active_pipeline = pipeline or _build_pipeline()
    try:
        return active_pipeline.filter_working_targets(ads)
    finally:
        if owns_pipeline:
            _close_pipeline(active_pipeline)


def _vision(
    ads: list[dict],
    *,
    pipeline: AdPipeline | None = None,
) -> list[dict]:
    owns_pipeline = pipeline is None
    active_pipeline = pipeline or _build_pipeline()
    try:
        return active_pipeline.enrich_missing_specs_from_images(ads)
    finally:
        if owns_pipeline:
            _close_pipeline(active_pipeline)


def _apply_benchmark(
    ads: list[dict],
    dataset_path: str | None,
    *,
    pipeline: AdPipeline | None = None,
) -> list[dict]:
    if not dataset_path:
        return ads

    dataset = CpuBenchmarkDataset.from_csv(dataset_path)

    # Сначала дешёвый локальный поиск. В Gemini отправляются только несовпавшие CPU.
    initially_enriched = dataset.enrich_ads(ads)
    unmatched_indexes = [
        index
        for index, ad in enumerate(initially_enriched)
        if ad.get("cpu_model") and "cpu_mark" not in ad
    ]
    unmatched = [initially_enriched[index] for index in unmatched_indexes]

    if unmatched:
        owns_pipeline = pipeline is None
        active_pipeline = pipeline or _build_pipeline()
        try:
            normalized = active_pipeline.normalize_cpu_models_for_benchmark(
                unmatched
            )
        finally:
            if owns_pipeline:
                _close_pipeline(active_pipeline)

        normalized_count = sum(
            1 for ad in normalized if ad.get("cpu_model_normalized") is True
        )
        if normalized_count:
            logger.info("AI нормализовал названия CPU: %s", normalized_count)

        enriched_unmatched = dataset.enrich_ads(normalized)
        result = list(initially_enriched)
        for index, enriched in zip(
            unmatched_indexes,
            enriched_unmatched,
            strict=False,
        ):
            result[index] = enriched
    else:
        result = initially_enriched

    matched = sum(1 for ad in result if "cpu_mark" in ad)
    logger.info("Benchmark найден для %s из %s объявлений", matched, len(result))
    return result
