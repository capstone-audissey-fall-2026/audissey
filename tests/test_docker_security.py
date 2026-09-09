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
            ui["ports"], ["127.0.0.1:${V2A_INSPECT_UI_PORT:-8501}:8501"]
        )
        self.assertLess(dockerfile.index("USER appuser"), dockerfile.index("CMD ["))


if __name__ == "__main__":
    unittest.main()
