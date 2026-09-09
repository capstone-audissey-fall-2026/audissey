from __future__ import annotations

import unittest

from v2a_inspect_server.app import app


class ServerAppTest(unittest.TestCase):
    def test_unused_vision_endpoints_are_not_registered(self) -> None:
        paths = {route.path for route in app.routes}

        self.assertNotIn("/infer/dinov2/embed-images", paths)
        self.assertNotIn("/infer/score", paths)


if __name__ == "__main__":
    unittest.main()
