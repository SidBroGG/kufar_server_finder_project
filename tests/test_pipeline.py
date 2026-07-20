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
                real_price=45,
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
            )
        ]


def test_filters_and_updates_price_without_mutating_source():
    source = [
        {"link": "https://example/working", "price": 50, "title": "PC"},
        {"link": "https://example/broken", "price": 20, "title": "Broken"},
    ]

    result = AdPipeline(FakeAnalyzer()).filter_working_targets(source)

    assert result == [
        {"link": "https://example/working", "price": 45.0, "title": "PC"}
    ]
    assert source[0]["price"] == 50


def test_extracts_only_explicit_specs_and_marks_source():
    source = [{"link": "https://example/working", "price": 50, "title": "PC"}]

    result = AdPipeline(FakeAnalyzer()).filter_working_targets(
        source,
        extract_specs=True,
    )

    assert result[0]["cpu_model"] == "Core i5-3470"
    assert result[0]["cpu_model_source"] == "text_exact"
    assert "ram_type" not in result[0]
    assert result[0]["ram_gb"] == 8
    assert result[0]["ram_gb_source"] == "text_exact"


def test_vision_fills_only_missing_values_and_marks_guesses():
    source = [
        {
            "link": "https://example/working",
            "cpu_model": "Core i5-3470",
            "cpu_model_source": "text_exact",
            "ram_type": None,
        }
    ]

    result = AdPipeline(FakeAnalyzer()).enrich_missing_specs_from_images(source)

    assert result[0]["cpu_model"] == "Core i5-3470"
    assert result[0]["cpu_model_source"] == "text_exact"
    assert result[0]["ram_type"] == "DDR3"
    assert result[0]["ram_type_source"] == "image_guess"
    assert result[0]["ram_type_confidence"] == "high"
    assert result[0]["ram_gb"] == 16
    assert result[0]["ram_gb_source"] == "image_guess"
    assert source[0]["ram_type"] is None
