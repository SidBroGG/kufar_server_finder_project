from kufar_server_finder.models import (
    AdAnalysis,
    CpuNameNormalization,
    VisionComponentSpec,
)
from kufar_server_finder.pipeline import AdPipeline


class FakeAnalyzer:
    def analyze_ads(self, ads):
        return [
            AdAnalysis(
                link="https://example/working",
                is_target=True,
                is_working=True,
                price=40,
                cpu_model="Core i5-3470",
                ram_gb=8,
            ),
            AdAnalysis(
                link="https://example/broken",
                is_target=True,
                is_working=False,
                price=20,
            ),
        ]

    def infer_specs_from_images(self, ads):
        return [
            VisionComponentSpec(
                link="https://example/working",
                cpu_model="Core i7-3770",
                cpu_model_confidence="low",
                ram_type="DDR3",
                ram_type_confidence="high",
                ram_gb=16,
                ram_gb_confidence="medium",
                cpu_socket=None,
                cpu_socket_confidence=None,
                product_type="thin_client",
                product_type_confidence="high",
                estimated_system_power_w=35,
                estimated_system_power_w_confidence="medium",
            )
        ]


def test_filters_and_updates_price_without_mutating_source():
    source = [
        {"link": "https://example/working", "price": 50},
        {"link": "https://example/broken", "price": 20},
    ]
    result = AdPipeline(FakeAnalyzer()).filter_working_targets(source)
    assert len(result) == 1
    assert result[0]["link"] == "https://example/working"
    assert result[0]["price"] == 40.0
    assert result[0]["cpu_model"] == "Core i5-3470"
    assert result[0]["ram_gb"] == 8
    assert source[0] == {"link": "https://example/working", "price": 50}


def test_missing_analysis_is_preserved_and_marked():
    class EmptyAnalyzer(FakeAnalyzer):
        def analyze_ads(self, ads):
            return []

    source = [{"link": "https://example/a", "price": 10}]
    result = AdPipeline(EmptyAnalyzer()).filter_working_targets(source)
    assert result[0]["analysis_status"] == "pending"
    assert "analysis_error" in result[0]


def test_description_network_failure_prevents_false_removal():
    source = [
        {
            "link": "https://example/broken",
            "price": 20,
            "description_status": "load_error",
            "description_load_error": True,
        }
    ]
    result = AdPipeline(FakeAnalyzer()).filter_working_targets(source)
    assert len(result) == 1
    assert result[0]["analysis_status"] == "pending"


def test_extracts_explicit_specs_by_default_and_marks_source():
    source = [{"link": "https://example/working", "price": 50, "title": "PC"}]
    result = AdPipeline(FakeAnalyzer()).filter_working_targets(source)
    assert result[0]["cpu_model"] == "Core i5-3470"
    assert result[0]["cpu_model_source"] == "text_exact"
    assert result[0]["ram_gb"] == 8
    assert result[0]["cpu_socket"] == "LGA1155"
    assert result[0]["cpu_socket_source"] == "cpu_model_guess"
    assert result[0]["cpu_socket_confidence"] == "high"


def test_vision_adds_product_type_and_estimated_system_power():
    source = [{"link": "https://example/working"}]

    result = AdPipeline(FakeAnalyzer()).enrich_missing_specs_from_images(source)

    assert result[0]["product_type"] == "thin_client"
    assert result[0]["product_type_source"] == "image_guess"
    assert result[0]["product_type_confidence"] == "high"
    assert result[0]["estimated_system_power_w"] == 35
    assert result[0]["estimated_system_power_w_source"] == "image_guess"
    assert result[0]["estimated_system_power_w_confidence"] == "medium"
    assert result[0]["cpu_socket"] == "LGA1155"
    assert result[0]["cpu_socket_source"] == "cpu_model_guess"
    assert result[0]["cpu_socket_confidence"] == "low"
    assert source == [{"link": "https://example/working"}]


def test_text_socket_guess_has_priority_over_cpu_model_mapping():
    class DescriptionSocketAnalyzer(FakeAnalyzer):
        def analyze_ads(self, ads):
            return [
                AdAnalysis(
                    link="https://example/working",
                    is_target=True,
                    is_working=True,
                    price=40,
                    cpu_model="Core i5-3470",
                    cpu_socket="LGA1150",
                    cpu_socket_source="description_guess",
                    cpu_socket_confidence="medium",
                )
            ]

    source = [{"link": "https://example/working", "price": 50}]
    result = AdPipeline(DescriptionSocketAnalyzer()).filter_working_targets(source)

    assert result[0]["cpu_socket"] == "LGA1150"
    assert result[0]["cpu_socket_source"] == "description_guess"


def test_vision_does_not_overwrite_existing_product_type_or_power():
    source = [
        {
            "link": "https://example/working",
            "product_type": "desktop_pc",
            "estimated_system_power_w": 120,
        }
    ]

    result = AdPipeline(FakeAnalyzer()).enrich_missing_specs_from_images(source)

    assert result[0]["product_type"] == "desktop_pc"
    assert result[0]["estimated_system_power_w"] == 120

def test_vision_fills_core_fields_when_ai_is_uncertain():
    class UncertainVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return [
                VisionComponentSpec(
                    link="https://example/working",
                    cpu_model=None,
                    cpu_model_confidence=None,
                    ram_type=None,
                    ram_type_confidence=None,
                    ram_gb=None,
                    ram_gb_confidence=None,
                    product_type="thin_client",
                    product_type_confidence="medium",
                )
            ]

    source = [{"link": "https://example/working"}]
    result = AdPipeline(UncertainVisionAnalyzer()).enrich_missing_specs_from_images(
        source
    )

    assert result[0]["cpu_model"] == (
        "Intel Atom/Celeron / AMD Embedded (примерное семейство)"
    )
    assert result[0]["ram_type"] == "DDR3/DDR4 (примерно)"
    assert result[0]["ram_gb"] == 4
    assert result[0]["cpu_model_source"] == "visual_fallback"
    assert result[0]["ram_type_source"] == "visual_fallback"
    assert result[0]["ram_gb_source"] == "visual_fallback"
    assert result[0]["cpu_model_confidence"] == "low"
    assert result[0]["ram_type_confidence"] == "low"
    assert result[0]["ram_gb_confidence"] == "low"


def test_vision_fills_core_fields_when_ai_returns_no_result():
    class EmptyVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return []

    source = [{"link": "https://example/working"}]
    result = AdPipeline(EmptyVisionAnalyzer()).enrich_missing_specs_from_images(source)

    assert result[0]["cpu_model"]
    assert result[0]["ram_type"]
    assert result[0]["ram_gb"] > 0
    assert result[0]["cpu_model_source"] == "visual_fallback"


def test_vision_fallback_uses_known_socket():
    class EmptyVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return []

    source = [
        {
            "link": "https://example/working",
            "cpu_socket": "AM4",
            "cpu_socket_source": "description_guess",
        }
    ]
    result = AdPipeline(EmptyVisionAnalyzer()).enrich_missing_specs_from_images(source)

    assert result[0]["cpu_model"] == "AMD Ryzen 1000-5000 (примерное семейство)"
    assert result[0]["ram_type"] == "DDR4 (примерно)"
    assert result[0]["ram_gb"] == 8



def test_vision_fills_socket_type_power_and_confidences_when_ai_returns_no_result():
    class EmptyVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return []

    source = [{"link": "https://example/working"}]
    result = AdPipeline(EmptyVisionAnalyzer()).enrich_missing_specs_from_images(source)
    ad = result[0]

    assert ad["cpu_socket"] == "Неизвестный сокет (примерно)"
    assert ad["product_type"] == "other"
    assert ad["estimated_system_power_w"] == 100
    assert ad["cpu_socket_source"] == "visual_fallback"
    assert ad["product_type_source"] == "visual_fallback"
    assert ad["estimated_system_power_w_source"] == "visual_fallback"
    assert ad["cpu_socket_confidence"] == "low"
    assert ad["product_type_confidence"] == "low"
    assert ad["estimated_system_power_w_confidence"] == "low"


def test_vision_uses_product_type_for_socket_and_power_fallbacks():
    class UncertainVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return [
                VisionComponentSpec(
                    link="https://example/working",
                    product_type="thin_client",
                    product_type_confidence="medium",
                )
            ]

    source = [{"link": "https://example/working"}]
    result = AdPipeline(UncertainVisionAnalyzer()).enrich_missing_specs_from_images(
        source
    )
    ad = result[0]

    assert ad["cpu_socket"] == "BGA (soldered, примерно)"
    assert ad["cpu_socket_confidence"] == "low"
    assert ad["product_type"] == "thin_client"
    assert ad["product_type_confidence"] == "medium"
    assert ad["estimated_system_power_w"] == 25
    assert ad["estimated_system_power_w_confidence"] == "low"


def test_vision_adds_missing_confidence_without_overwriting_values():
    class EmptyVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return []

    source = [
        {
            "link": "https://example/working",
            "cpu_socket": "AM4",
            "cpu_socket_source": "text_exact",
            "product_type": "desktop_pc",
            "estimated_system_power_w": 120,
        }
    ]
    result = AdPipeline(EmptyVisionAnalyzer()).enrich_missing_specs_from_images(source)
    ad = result[0]

    assert ad["cpu_socket"] == "AM4"
    assert ad["product_type"] == "desktop_pc"
    assert ad["estimated_system_power_w"] == 120
    assert ad["cpu_socket_confidence"] == "high"
    assert ad["product_type_confidence"] == "low"
    assert ad["estimated_system_power_w_confidence"] == "low"


def test_vision_refines_incomplete_text_cpu_from_clear_photo():
    class ExactCpuVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return [
                VisionComponentSpec(
                    link="https://example/working",
                    cpu_model="Intel Atom N570",
                    cpu_model_confidence="high",
                )
            ]

    source = [
        {
            "link": "https://example/working",
            "cpu_model": "Intel Atom",
            "cpu_model_source": "text_exact",
            "cpu_mark": 1349,
            "cpu_benchmark_name": "Intel Atom x7-Z8750 @ 1.60GHz",
            "cpu_benchmark_source": "dataset",
            "ram_type": "DDR3",
            "ram_type_source": "text_exact",
            "ram_gb": 2,
            "ram_gb_source": "text_exact",
            "cpu_socket": "BGA (soldered)",
            "cpu_socket_source": "description_guess",
            "cpu_socket_confidence": "medium",
            "product_type": "laptop",
            "product_type_source": "image_guess",
            "product_type_confidence": "high",
            "estimated_system_power_w": 40,
            "estimated_system_power_w_source": "image_guess",
            "estimated_system_power_w_confidence": "medium",
        }
    ]

    result = AdPipeline(ExactCpuVisionAnalyzer()).enrich_missing_specs_from_images(
        source
    )
    ad = result[0]

    assert ad["cpu_model"] == "Intel Atom N570"
    assert ad["cpu_model_source"] == "image_guess"
    assert ad["cpu_model_confidence"] == "high"
    assert "cpu_mark" not in ad
    assert "cpu_benchmark_name" not in ad
    assert "cpu_benchmark_source" not in ad


def test_vision_does_not_replace_complete_text_cpu_even_with_high_confidence():
    class DifferentCpuVisionAnalyzer(FakeAnalyzer):
        def infer_specs_from_images(self, ads):
            return [
                VisionComponentSpec(
                    link="https://example/working",
                    cpu_model="Intel Core i7-3770",
                    cpu_model_confidence="high",
                )
            ]

    source = [
        {
            "link": "https://example/working",
            "cpu_model": "Intel Core i5-3470",
            "cpu_model_source": "text_exact",
        }
    ]
    result = AdPipeline(DifferentCpuVisionAnalyzer()).enrich_missing_specs_from_images(
        source
    )

    assert result[0]["cpu_model"] == "Intel Core i5-3470"
    assert result[0]["cpu_model_source"] == "text_exact"

def test_cpu_name_is_normalized_before_benchmark_without_changing_model_number():
    class NormalizingAnalyzer(FakeAnalyzer):
        def normalize_cpu_names(self, ads):
            return [
                CpuNameNormalization(
                    link="https://example/working",
                    normalized_cpu_model="AMD Athlon II X4 640",
                )
            ]

    source = [
        {
            "link": "https://example/working",
            "cpu_model": "athlone x4 640",
            "cpu_model_source": "text_exact",
        }
    ]
    result = AdPipeline(NormalizingAnalyzer()).normalize_cpu_models_for_benchmark(
        source
    )

    assert result[0]["cpu_model"] == "AMD Athlon II X4 640"
    assert result[0]["cpu_model_original"] == "athlone x4 640"
    assert result[0]["cpu_model_normalization_source"] == "gemini"
    assert source[0]["cpu_model"] == "athlone x4 640"


def test_cpu_name_normalization_rejects_changed_model_number():
    class UnsafeNormalizer(FakeAnalyzer):
        def normalize_cpu_names(self, ads):
            return [
                CpuNameNormalization(
                    link="https://example/working",
                    normalized_cpu_model="Intel Core 2 Quad Q6700",
                )
            ]

    source = [
        {
            "link": "https://example/working",
            "cpu_model": "Intel core 2 quad q6600",
        }
    ]
    result = AdPipeline(UnsafeNormalizer()).normalize_cpu_models_for_benchmark(
        source
    )

    assert result[0]["cpu_model"] == "Intel core 2 quad q6600"
    assert "cpu_model_normalized" not in result[0]


def test_filter_uses_single_analysis_call():
    class CombinedAnalyzer(FakeAnalyzer):
        def __init__(self):
            self.analysis_calls = 0

        def analyze_ads(self, ads):
            self.analysis_calls += 1
            return [
                AdAnalysis(
                    link="https://example/working",
                    is_target=True,
                    is_working=True,
                    price=10,
                    cpu_model="Intel Core i5-3470",
                    ram_type="DDR3",
                    ram_gb=8,
                )
            ]

    analyzer = CombinedAnalyzer()
    result = AdPipeline(analyzer).filter_working_targets(
        [{"link": "https://example/working", "price": 12}]
    )

    assert analyzer.analysis_calls == 1
    assert result[0]["cpu_model"] == "Intel Core i5-3470"
    assert result[0]["ram_type"] == "DDR3"
    assert result[0]["ram_gb"] == 8



def test_legacy_specs_pipeline_is_not_exposed():
    from kufar_server_finder import models, prompts

    pipeline = AdPipeline(FakeAnalyzer())
    assert not hasattr(pipeline, "_merge_explicit_specs")
    assert not hasattr(FakeAnalyzer(), "extract_explicit_specs")
    assert not hasattr(models, "PCComponentSpec")
    assert not hasattr(prompts, "SPECS_SYSTEM_INSTRUCTION")


def test_pipeline_context_manager_and_empty_filter_close_analyzer():
    class ClosableAnalyzer(FakeAnalyzer):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    analyzer = ClosableAnalyzer()
    with AdPipeline(analyzer) as pipeline:
        assert pipeline.filter_working_targets([]) == []

    assert analyzer.closed is True


def test_cpu_normalization_skips_ineligible_missing_and_unchanged_results():
    class Normalizer(FakeAnalyzer):
        def normalize_cpu_names(self, ads):
            return [
                CpuNameNormalization(
                    link="same",
                    normalized_cpu_model="Intel Core i5-3470",
                ),
                CpuNameNormalization(
                    link="empty",
                    normalized_cpu_model=None,
                ),
            ]

    source = [
        {"link": "none", "cpu_model": None},
        {
            "link": "fallback",
            "cpu_model": "CPU 1234",
            "cpu_model_source": "visual_fallback",
        },
        {"link": "missing", "cpu_model": "CPU 5678"},
        {"link": "same", "cpu_model": "Intel Core i5-3470"},
        {"link": "empty", "cpu_model": "CPU 9999"},
    ]

    pipeline = AdPipeline(Normalizer())
    assert pipeline.normalize_cpu_models_for_benchmark(
        [{"link": "none", "cpu_model": None}]
    ) == [{"link": "none", "cpu_model": None}]

    result = pipeline.normalize_cpu_models_for_benchmark(source)

    assert result == source


def test_visual_fallback_handles_invalid_numeric_value():
    ad = {"ram_gb": "invalid"}

    AdPipeline._set_visual_fallback(ad, "ram_gb", 8)

    assert ad["ram_gb"] == 8
    assert ad["ram_gb_source"] == "visual_fallback"


def test_exact_socket_removes_stale_confidence():
    ad = {"cpu_socket_confidence": "low"}

    AdPipeline._set_text_socket(ad, "AM4", "text_exact", "high")

    assert ad["cpu_socket"] == "AM4"
    assert "cpu_socket_confidence" not in ad


def test_changed_cpu_removes_derived_socket_and_benchmark():
    ad = {
        "cpu_model": "Intel Atom",
        "cpu_model_source": "image_guess",
        "cpu_model_confidence": "low",
        "cpu_socket": "BGA (soldered)",
        "cpu_socket_source": "cpu_model_guess",
        "cpu_socket_confidence": "low",
        "cpu_mark": 10,
        "cpu_benchmark_name": "Old CPU",
        "cpu_benchmark_source": "dataset",
    }

    AdPipeline._set_vision_guess(
        ad,
        "cpu_model",
        "Intel Atom N570",
        "high",
    )

    assert ad["cpu_model"] == "Intel Atom N570"
    assert "cpu_socket" not in ad
    assert "cpu_mark" not in ad


def test_cpu_normalization_eligibility_rejects_empty_and_visual_fallback():
    from kufar_server_finder.pipeline import _needs_cpu_name_normalization

    assert not _needs_cpu_name_normalization({"cpu_model": "unknown"})
    assert not _needs_cpu_name_normalization(
        {"cpu_model": "CPU 1234", "cpu_model_source": "visual_fallback"}
    )
