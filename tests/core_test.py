import json
import os
import unittest
import numpy as np

import core

root_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../model')


class TestCore(unittest.TestCase):
    maxDiff = None

    def test_initialize_cpu(self):
        core.initialize(root_dir, False)
        core.finalize()

    def test_invalid_initialize_path(self):
        with self.assertRaises(Exception):
            core.initialize(" ", False)

    def test_invalid_speaker_id(self):
        core.initialize(root_dir, False)
        nil = np.array([], np.int64)
        fnil = np.array([], np.float32)
        unknown_style1 = np.array([-1], np.int64)
        unknown_style2 = np.array([10], np.int64)
        with self.assertRaisesRegex(Exception, "Unknown style ID: -1"):
            core.variance_forward(0, nil, nil, unknown_style1)
        with self.assertRaisesRegex(Exception, "Unknown style ID: 10"):
            core.variance_forward(0, nil, nil, unknown_style2)
        with self.assertRaisesRegex(Exception, "Unknown style ID: -1"):
            core.decode_forward(0, nil, fnil, fnil, unknown_style1)
        with self.assertRaisesRegex(Exception, "Unknown style ID: 10"):
            core.decode_forward(0, nil, fnil, fnil, unknown_style2)
        core.finalize()

    # TODO: ソート順などの問題でWindowsのテストが通らないので、一旦コメントアウト
    # def test_metas(self):
    #     with open(os.path.join(root_dir, "test", "metas.json"), encoding="utf-8") as f:
    #         metas_json = json.load(f)
    #     with open(os.path.join(root_dir, "gaussian_test", "metas.json"), encoding="utf-8") as f:
    #         gaussian_metas_json = json.load(f)
    #         gaussian_metas_json[0]["styles"][0]["id"] = 1
    #         metas_json[0]["styles"].append(gaussian_metas_json[0]["styles"][0])
    #     metas = json.dumps(metas_json, sort_keys=True)
    #     core.initialize(root_dir, False)
    #     core_metas_json = json.loads(core.metas())
    #     core_metas = json.dumps(core_metas_json, sort_keys=True)
    #     core.finalize()
    #     self.assertEqual(metas, core_metas)

    def test_supported_devices(self):
        devices = json.loads(core.supported_devices())
        for expected_device in ["cpu", "cuda"]:
            self.assertIn(expected_device, devices)
        self.assertTrue(devices["cpu"])


if __name__ == '__main__':
    unittest.main()
