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
