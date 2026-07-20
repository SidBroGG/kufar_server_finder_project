from kufar_server_finder.models import AdAnalysis, PCComponentSpec
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

    def infer_specs(self, ads):
        return [
            PCComponentSpec(
                link="https://example/working",
                cpu_model="Core i5-3470",
                ram_type="DDR3",
                ram_gb=8,
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


def test_can_enrich_specs():
    source = [{"link": "https://example/working", "price": 50, "title": "PC"}]

    result = AdPipeline(FakeAnalyzer()).filter_working_targets(
        source, infer_specs=True
    )

    assert result[0]["cpu_model"] == "Core i5-3470"
    assert result[0]["ram_type"] == "DDR3"
    assert result[0]["ram_gb"] == 8
