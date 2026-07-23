from kufar_server_finder.prompts import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    SPECS_SYSTEM_INSTRUCTION,
    VISION_SPECS_SYSTEM_INSTRUCTION,
)


def test_analysis_prompt_uses_minimum_debian_configuration_rules():
    prompt = ANALYSIS_SYSTEM_INSTRUCTION

    assert "Debian-сервер" in prompt
    assert "видеокарту-затычку" in prompt
    assert "матрица" in prompt
    assert "работать из коробки" in prompt
    assert "real_price=80" in prompt
    assert "ram_gb=4" in prompt
    assert "minimum_configuration" in prompt
    assert "price_components" in prompt
    assert "характеристики более дорогой опции" in prompt


def test_specs_and_vision_prompts_keep_selected_minimum_configuration():
    assert "самой дешёвой" in SPECS_SYSTEM_INSTRUCTION
    assert "real_price=80" in SPECS_SYSTEM_INSTRUCTION
    assert "характеристиками более дорогой опции" in VISION_SPECS_SYSTEM_INSTRUCTION
