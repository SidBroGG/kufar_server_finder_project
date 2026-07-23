from kufar_server_finder.visual_refinement import (
    cpu_model_specificity,
    fields_needing_visual_analysis,
    should_replace_with_vision,
)


def test_incomplete_cpu_is_sent_to_visual_analysis():
    ad = {
        "cpu_model": "Intel Atom",
        "cpu_model_source": "text_exact",
        "cpu_socket": "BGA (soldered)",
        "ram_type": "DDR3",
        "ram_gb": 2,
        "product_type": "laptop",
        "estimated_system_power_w": 40,
    }

    assert "cpu_model" in fields_needing_visual_analysis(ad)


def test_exact_cpu_is_not_sent_for_refinement():
    ad = {
        "cpu_model": "Intel Atom x7-Z8750",
        "cpu_model_source": "text_exact",
        "cpu_socket": "BGA (soldered)",
        "ram_type": "DDR3",
        "ram_gb": 2,
        "product_type": "laptop",
        "estimated_system_power_w": 40,
    }

    assert "cpu_model" not in fields_needing_visual_analysis(ad)
    assert cpu_model_specificity("Intel Atom x7-Z8750") == 2


def test_text_family_requires_high_confidence_and_more_specific_model():
    ad = {"cpu_model": "Intel Atom", "cpu_model_source": "text_exact"}

    assert should_replace_with_vision(ad, "cpu_model", "Intel Atom N570", "high")
    assert not should_replace_with_vision(
        ad, "cpu_model", "Intel Atom N570", "medium"
    )
    assert not should_replace_with_vision(ad, "cpu_model", "Intel Atom", "high")

import pytest


@pytest.mark.parametrize(
    "field",
    [
        "cpu_model",
        "cpu_socket",
        "ram_type",
        "ram_gb",
        "product_type",
        "estimated_system_power_w",
    ],
)
def test_missing_visual_fields_are_requested(field):
    assert field in fields_needing_visual_analysis({})


def test_low_confidence_fallbacks_and_generic_values_are_requested():
    ad = {
        "cpu_model": "Intel Core family",
        "cpu_model_source": "visual_fallback",
        "cpu_model_confidence": "low",
        "cpu_socket": "LGA115x",
        "ram_type": "DDR3/DDR4",
        "ram_gb": 8,
        "product_type": "other",
        "estimated_system_power_w": 100,
    }
    fields = fields_needing_visual_analysis(ad)
    assert "cpu_model" in fields
    assert "cpu_socket" in fields
    assert "ram_type" in fields
    assert "product_type" in fields


def test_vision_replacement_rules_cover_confidence_and_specificity():
    assert not should_replace_with_vision({}, "cpu_model", None, "high")
    assert should_replace_with_vision({}, "cpu_model", "Intel Core i5-3470", "low")
    assert not should_replace_with_vision(
        {"cpu_model": "Intel Core i5-3470"},
        "cpu_model",
        "Intel Core i5-3470",
        "high",
    )
    assert should_replace_with_vision(
        {"cpu_model": "family", "cpu_model_source": "visual_fallback"},
        "cpu_model",
        "Intel Core i5-3470",
        "medium",
    )
    assert should_replace_with_vision(
        {"product_type": "other"},
        "product_type",
        "desktop_pc",
        "medium",
    )
    assert should_replace_with_vision(
        {"cpu_model": "family", "cpu_model_confidence": "low"},
        "cpu_model",
        "Intel Core i5-3470",
        "medium",
    )
    assert should_replace_with_vision(
        {
            "ram_gb": 8,
            "ram_gb_source": "image_guess",
            "ram_gb_confidence": "low",
        },
        "ram_gb",
        16,
        "medium",
    )
    assert not should_replace_with_vision(
        {"ram_gb": 8}, "ram_gb", 16, "low"
    )


def test_cpu_specificity_handles_empty_approximate_and_exact_values():
    assert cpu_model_specificity(None) == 0
    assert cpu_model_specificity("Intel family примерно") == 1
    assert cpu_model_specificity("Xeon E5-2670 v2") == 2


def test_low_confidence_non_text_values_are_requested():
    ad = {
        "cpu_model": "Intel Core i5-3470",
        "cpu_model_source": "image_guess",
        "cpu_model_confidence": "low",
        "cpu_socket": "LGA1155",
        "ram_type": "DDR3",
        "ram_gb": 8,
        "product_type": "desktop_pc",
        "estimated_system_power_w": 100,
    }

    assert "cpu_model" in fields_needing_visual_analysis(ad)


def test_specificity_helpers_cover_empty_and_known_fields():
    from kufar_server_finder.visual_refinement import (
        _field_specificity,
        _ram_type_specificity,
        _socket_specificity,
    )

    assert _field_specificity("ram_type", "DDR4") == 2
    assert _field_specificity("cpu_socket", "AM4") == 2
    assert _ram_type_specificity(None) == 0
    assert _socket_specificity(None) == 0
    assert _socket_specificity("примерный сокет") == 1
