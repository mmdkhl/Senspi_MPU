import pathlib
import sys
import unittest

# Ensure src/ is on path for direct test execution
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sensepi.config.sampling import SamplingConfig
from sensepi.config.pi_logger_config import PiLoggerConfig


class SamplingSingleRateTest(unittest.TestCase):
    def test_sampling_aliases_are_equal(self):
        cfg = SamplingConfig(device_rate_hz=200.0)
        self.assertEqual(cfg.device_rate_hz, cfg.record_rate_hz)
        self.assertEqual(cfg.device_rate_hz, cfg.stream_rate_hz)
        self.assertEqual(cfg.record_decimate, 1)
        self.assertEqual(cfg.stream_decimate, 1)

    def test_pi_logger_config_invariants(self):
        sampling = SamplingConfig(device_rate_hz=123.0)
        pi_cfg = PiLoggerConfig.from_sampling(sampling)
        data = pi_cfg.to_pi_config_dict()
        self.assertEqual(data["device_rate_hz"], sampling.device_rate_hz)
        self.assertEqual(data["record_decimate"], 1)
        self.assertEqual(data["stream_decimate"], 1)
        self.assertNotIn("record_rate_hz", data)
        self.assertNotIn("stream_rate_hz", data)

    def test_sampling_config_prefers_top_level_block(self):
        payload = {
            "sampling": {"device_rate_hz": 200, "mode": "high_fidelity"},
            "sensors": {"mpu6050": {"sample_rate_hz": 512}},
        }
        cfg = SamplingConfig.from_mapping(payload)
        self.assertEqual(cfg.device_rate_hz, 200)
        self.assertEqual(cfg.mode_key, "high_fidelity")

    def test_sampling_config_defaults_to_200hz(self):
        cfg = SamplingConfig.from_mapping(None)
        self.assertEqual(cfg.device_rate_hz, 200.0)

    def test_sampling_config_ignores_legacy_sensor_sample_rate(self):
        payload = {"sensors": {"mpu6050": {"sample_rate_hz": 512}}}
        cfg = SamplingConfig.from_mapping(payload)
        self.assertEqual(cfg.device_rate_hz, 200.0)


if __name__ == "__main__":
    unittest.main()
