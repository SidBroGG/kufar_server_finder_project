from openpyxl import load_workbook

from kufar_finder_core import save_items
from kufar_server_finder.excel_export import export_ads_json_to_excel


def test_export_ads_json_to_excel(tmp_path):
    json_path = tmp_path / "output.json"
    excel_path = tmp_path / "output.xlsx"
    save_items(
        json_path,
        [
            {
                "product_type": "thin_client",
                "product_type_confidence": "high",
                "price": 25.5,
                "ram_type": "DDR3",
                "ram_type_confidence": "medium",
                "ram_gb": None,
                "cpu_socket": "LGA1155",
                "cpu_socket_confidence": "low",
                "cpu_model": "Intel Core i5-3470",
                "cpu_model_source": "text_exact",
                "cpu_mark": 4665,
                "cpu_benchmark_source": "dataset",
                "estimated_system_power_w": 60,
                "estimated_system_power_w_confidence": "medium",
                "link": "https://example.test/item/1",
            }
        ],
    )

    export_ads_json_to_excel(json_path, excel_path)

    workbook = load_workbook(excel_path)
    sheet = workbook["Объявления"]

    assert [cell.value for cell in sheet[1]] == [
        "Тип устройства",
        "Цена, BYN",
        "Тип ОЗУ",
        "Объём ОЗУ, ГБ",
        "Сокет процессора",
        "Модель процессора",
        "CPU Benchmark",
        "Оценочная мощность системы, Вт",
        "Ссылка",
    ]
    assert sheet["A2"].value == "Тонкий клиент"
    assert sheet["B2"].value == 25.5
    assert sheet["D2"].value == "—"
    assert sheet["I2"].hyperlink.target == "https://example.test/item/1"

    assert sheet["A2"].fill.fgColor.rgb == "00C6EFCE"
    assert sheet["C2"].fill.fgColor.rgb == "00FCE4D6"
    assert sheet["E2"].fill.fgColor.rgb == "00FFC7CE"
    assert sheet["F2"].fill.fgColor.rgb == "00C6EFCE"
    assert sheet["G2"].fill.fgColor.rgb == "00C6EFCE"
    assert sheet["H2"].fill.fgColor.rgb == "00FCE4D6"


def test_excel_helpers_handle_non_links_and_missing_product_type():
    from openpyxl import Workbook
    from kufar_server_finder.excel_export import (
        MISSING_VALUE,
        _display_product_type,
        _format_link,
    )

    cell = Workbook().active["A1"]
    _format_link(cell, "not-a-link")

    assert cell.hyperlink is None
    assert _display_product_type(None) == MISSING_VALUE
