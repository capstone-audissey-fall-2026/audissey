from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DockerSecurityTest(unittest.TestCase):
    def test_inference_is_private_and_non_root(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yaml").read_text())
        inference = compose["services"]["inference"]
        ui = compose["services"]["v2a-inspect"]
        dockerfile = (ROOT / "server" / "Dockerfile").read_text()

        self.assertNotIn("ports", inference)
        self.assertEqual(
            inference["gpus"],
            [{"driver": "nvidia", "device_ids": ["${V2A_GPU_DEVICE:-0}"]}],
        )
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", inference["environment"])
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES=all", dockerfile)
        self.assertEqual(
            ui["ports"], ["127.0.0.1:${V2A_INSPECT_UI_PORT:-8501}:8501"]
        )
        self.assertLess(dockerfile.index("USER appuser"), dockerfile.index("CMD ["))


    def test_hunyuan_uses_the_only_visible_gpu(self) -> None:
        source = (
            ROOT / "server/src/v2a_inspect_server/inference/hunyuan.py"
        ).read_text()

        self.assertNotIn("cuda:1", source)


if __name__ == "__main__":
    unittest.main()
