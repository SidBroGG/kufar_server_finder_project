from kufar_server_finder.models import (
    AdAnalysis,
    PCComponentSpec,
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
                real_price=40,
            ),
            AdAnalysis(
                link="https://example/broken",
                is_target=True,
                is_working=False,
                real_price=20,
            ),
        ]

    def extract_explicit_specs(self, ads):
        return [
            PCComponentSpec(
                link="https://example/working",
                cpu_model="Core i5-3470",
                ram_type=None,
                ram_gb=8,
                cpu_socket=None,
            )
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
    assert result == [{"link": "https://example/working", "price": 40.0}]
    assert source[0]["price"] == 50


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


def test_extracts_explicit_specs_and_marks_source():
    source = [{"link": "https://example/working", "price": 50, "title": "PC"}]
    result = AdPipeline(FakeAnalyzer()).filter_working_targets(
        source,
        extract_specs=True,
    )
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
        def extract_explicit_specs(self, ads):
            return [
                PCComponentSpec(
                    link="https://example/working",
                    cpu_model="Core i5-3470",
                    ram_type=None,
                    ram_gb=None,
                    cpu_socket="LGA1150",
                    cpu_socket_source="description_guess",
                    cpu_socket_confidence="medium",
                )
            ]

    source = [{"link": "https://example/working", "price": 50}]
    result = AdPipeline(DescriptionSocketAnalyzer()).filter_working_targets(
        source,
        extract_specs=True,
    )

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
