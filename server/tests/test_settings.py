from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect_server.settings import ServerSettings


class ServerSettingsTest(unittest.TestCase):
    def test_hunyuan_model_path_uses_server_environment(self) -> None:
        with patch.dict(
            os.environ,
            {"V2A_SERVER_HUNYUAN_MODEL_PATH": "/tmp/v2a-model-cache"},
        ):
            settings = ServerSettings(_env_file=None)

        self.assertEqual(settings.hunyuan_model_path, Path("/tmp/v2a-model-cache"))


if __name__ == "__main__":
    unittest.main()
