from kufar_server_finder.socket_inference import infer_socket_from_cpu


def test_infers_common_desktop_sockets():
    assert infer_socket_from_cpu("Intel Core i5-3470").socket == "LGA1155"
    assert infer_socket_from_cpu("Intel Core i5-10400F").socket == "LGA1200"
    assert infer_socket_from_cpu("Intel Core i5-12400").socket == "LGA1700"
    assert infer_socket_from_cpu("AMD Ryzen 5 3600").socket == "AM4"
    assert infer_socket_from_cpu("AMD Ryzen 5 7600").socket == "AM5"


def test_infers_server_and_mobile_packages():
    assert infer_socket_from_cpu("Xeon E5-2670 v2").socket == "LGA2011"
    assert infer_socket_from_cpu("Xeon E5-2680 v4").socket == "LGA2011-3"
    assert infer_socket_from_cpu("Intel Core i7-3630QM").socket == "rPGA988B"
    assert infer_socket_from_cpu("Intel Core i5-8250U").socket == "BGA (soldered)"


def test_unknown_cpu_returns_none():
    assert infer_socket_from_cpu("неизвестный процессор") is None

import pytest


@pytest.mark.parametrize(
    ("cpu", "socket", "confidence"),
    [
        ("Socket LGA 1155", "LGA1155", "high"),
        ("socket strx4", "sTRX4", "high"),
        ("AMD Threadripper 1950X", "TR4", "high"),
        ("AMD Threadripper 3970X", "sTRX4", "high"),
        ("AMD Threadripper 7980X", "sTR5", "high"),
        ("AMD EPYC 7452", "SP3", "medium"),
        ("AMD EPYC 8534", "SP6", "medium"),
        ("AMD EPYC 9654", "SP5", "medium"),
        ("Xeon E3-1230", "LGA1155", "high"),
        ("Xeon E3-1231 v3", "LGA1150", "high"),
        ("Xeon E3-1240 v5", "LGA1151", "high"),
        ("Xeon E-2136", "LGA1151", "medium"),
        ("Xeon E-2336", "LGA1200", "medium"),
        ("Xeon E-2436", "LGA1700", "medium"),
        ("Xeon W-1290", "LGA1200", "medium"),
        ("Xeon W-2295", "LGA2066", "medium"),
        ("Xeon W-2495", "LGA4677", "medium"),
        ("Xeon Gold 6130", "LGA3647", "medium"),
        ("Xeon Gold 6330", "LGA4189", "medium"),
        ("Xeon Gold 6430", "LGA4677", "medium"),
        ("Intel Core Ultra 7 265K", "LGA1851", "medium"),
        ("Intel Core i7-980X", "LGA1366", "high"),
        ("Intel Core i7-3960X", "LGA2011", "high"),
        ("Intel Core i7-5820K", "LGA2011-3", "high"),
        ("Intel Core i9-10980XE", "LGA2066", "high"),
        ("Intel Core i7-4700MQ", "rPGA947", "medium"),
        ("Intel Core i7-7700HQ", "BGA (soldered)", "medium"),
        ("AMD Ryzen 7 7840U", "BGA (soldered)", "low"),
        ("AMD FX-8350", "AM3+", "high"),
        ("AMD Phenom II X4 955", "AM3", "medium"),
        ("AMD Athlon 3000G", "AM4", "high"),
        ("Pentium G695", "LGA1156", "low"),
        ("Pentium G2020", "LGA1155", "medium"),
        ("Pentium G3258", "LGA1150", "medium"),
        ("Celeron G4900", "LGA1151", "medium"),
        ("Celeron G5900", "LGA1151", "medium"),
        ("Celeron G6900", "LGA1200", "medium"),
        ("Celeron G7900", "LGA1700", "medium"),
    ],
)
def test_infers_extended_cpu_families(cpu, socket, confidence):
    result = infer_socket_from_cpu(cpu)
    assert result is not None
    assert result.socket == socket
    assert result.confidence == confidence


def test_blank_and_unmapped_models_return_none():
    assert infer_socket_from_cpu(None) is None
    assert infer_socket_from_cpu("   ") is None
    assert infer_socket_from_cpu("Threadripper unknown") is None
    assert infer_socket_from_cpu("Pentium G9000") is None


def test_intel_generation_rejects_unexpected_model_length():
    from kufar_server_finder.socket_inference import _intel_generation

    assert _intel_generation("123") is None
