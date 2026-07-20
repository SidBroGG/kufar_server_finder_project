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
