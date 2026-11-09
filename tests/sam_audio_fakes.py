"""Small doubles for the pinned SAM Audio tensor and device boundaries."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()


def fake_runtime(model_name, processor_name):
    torch = SimpleNamespace(
        from_numpy=Tensor, inference_mode=nullcontext, manual_seed=Mock()
    )
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    batch = Mock()
    batch.to.return_value = batch
    processor = Mock(return_value=batch)
    sam = SimpleNamespace(**{
        model_name: SimpleNamespace(from_pretrained=Mock(return_value=model)),
        processor_name: SimpleNamespace(from_pretrained=Mock(return_value=processor)),
    })
    return torch, sam, model, processor, batch
