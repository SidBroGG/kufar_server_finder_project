from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .benchmark import CpuBenchmarkDataset
from .config import GeminiConfig, KufarConfig
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
    _add_extract_specs_argument(analyze)

    vision = subparsers.add_parser(
        "vision",
        help="Отдельно угадать отсутствующие характеристики по фотографиям",
    )
    vision.add_argument("--input", default="output.json")
    vision.add_argument("--output", default="output_vision.json")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Добавить CPU benchmark points из CSV",
    )
    benchmark.add_argument("--input", default="output_vision.json")
    benchmark.add_argument("--output", default="output_benchmark.json")
    benchmark.add_argument("--dataset", default="CPU_benchmark_v4.csv")

    run = subparsers.add_parser("run", help="Собрать объявления и сразу обработать")
    _add_collect_arguments(run)
    run.add_argument("--raw-output", default="output_unfiltered.json")
    run.add_argument("--output", default="output.json")
    _add_extract_specs_argument(run)

    return parser


def _add_extract_specs_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extract-specs",
        "--infer-specs",
        dest="extract_specs",
        action="store_true",
        help="Извлечь только явно написанные CPU и ОЗУ, без догадок",
    )


def _add_collect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", default=None)
    parser.add_argument("--computers-only", action="store_true")
    parser.add_argument("--max-price", type=float, default=100.0)
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="Не открывать каждое объявление для загрузки описания",
    )
    parser.add_argument("--region", default="7")
    parser.add_argument("--page-delay", type=float, default=1.0)
    parser.add_argument("--detail-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)


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
            result = _analyze(ads, extract_specs=args.extract_specs)
            save_ads(args.output, result)
            logger.info("Результат сохранён: %s", args.output)
            return 0

        if args.command == "vision":
            ads = load_ads(args.input)
            result = _vision(ads)
            save_ads(args.output, result)
            logger.info("Результат фото-анализа сохранён: %s", args.output)
            return 0

        if args.command == "benchmark":
            ads = load_ads(args.input)
            result = _benchmark(ads, dataset_path=args.dataset)
            save_ads(args.output, result)
            logger.info("Результат benchmark сохранён: %s", args.output)
            return 0

        if args.command == "run":
            ads = _collect(args)
            save_ads(args.raw_output, ads)
            result = _analyze(ads, extract_specs=args.extract_specs)
            result = _vision(result)
            result = _benchmark(ads=result, dataset_path=args.dataset)
            save_ads(args.output, result)
            logger.info(
                "Сырые данные: %s; итог: %s",
                args.raw_output,
                args.output,
            )
            return 0
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Неожиданная ошибка")
        return 1

    return 1


def _collect(args: argparse.Namespace) -> list[dict]:
    config = KufarConfig(
        region=args.region,
        request_timeout=args.timeout,
        page_delay=max(args.page_delay, 0),
        detail_delay=max(args.detail_delay, 0),
    )
    return KufarClient(config).fetch_ads(
        query=args.query,
        computers_only=args.computers_only,
        max_price=args.max_price,
        load_descriptions=not args.no_descriptions,
    )


def _build_pipeline() -> AdPipeline:
    config = GeminiConfig.from_env()
    from .gemini import GeminiAnalyzer

    return AdPipeline(GeminiAnalyzer(config))


def _analyze(ads: list[dict], *, extract_specs: bool) -> list[dict]:
    return _build_pipeline().filter_working_targets(
        ads,
        extract_specs=extract_specs,
    )


def _vision(ads: list[dict]) -> list[dict]:
    return _build_pipeline().enrich_missing_specs_from_images(ads)


def _benchmark(
    ads: list[dict],
    *,
    dataset_path: str,
) -> list[dict]:
    dataset = CpuBenchmarkDataset.from_csv(dataset_path)
    return dataset.add_points(ads)
