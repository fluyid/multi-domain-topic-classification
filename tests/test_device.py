from topicclf.device import get_device

def test_get_device_returns_a_known_device():
    assert get_device() in {"cuda", "mps", "cpu"}