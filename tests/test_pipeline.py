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


def test_vision_adds_product_type_and_estimated_system_power():
    source = [{"link": "https://example/working"}]

    result = AdPipeline(FakeAnalyzer()).enrich_missing_specs_from_images(source)

    assert result[0]["product_type"] == "thin_client"
    assert result[0]["product_type_source"] == "image_guess"
    assert result[0]["product_type_confidence"] == "high"
    assert result[0]["estimated_system_power_w"] == 35
    assert result[0]["estimated_system_power_w_source"] == "image_guess"
    assert result[0]["estimated_system_power_w_confidence"] == "medium"
    assert source == [{"link": "https://example/working"}]


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
