from kufar_server_finder.prompts import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    SPECS_SYSTEM_INSTRUCTION,
    VISION_SPECS_SYSTEM_INSTRUCTION,
)


def test_analysis_prompt_uses_selected_configuration_fields():
    prompt = ANALYSIS_SYSTEM_INSTRUCTION

    assert "Debian-сервер" in prompt
    assert "видеокарту-затычку" in prompt
    assert "матрица" in prompt
    assert "работать из коробки" in prompt
    assert "price=80" in prompt
    assert "ram_gb=4" in prompt
    assert "cpu_model, ram_type и ram_gb" in prompt
    assert "характеристики более дорогой опции" in prompt
    assert "minimum_configuration" not in prompt
    assert "price_components" not in prompt
    assert "real_price" not in prompt


def test_specs_and_vision_prompts_keep_selected_configuration():
    assert "самой дешёвой" in SPECS_SYSTEM_INSTRUCTION
    assert "price=80" in SPECS_SYSTEM_INSTRUCTION
    assert "cpu_model, ram_type и ram_gb" in SPECS_SYSTEM_INSTRUCTION
    assert "характеристики более дорогой опции" in VISION_SPECS_SYSTEM_INSTRUCTION
