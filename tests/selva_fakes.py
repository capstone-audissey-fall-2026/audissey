"""CPU doubles for the pinned SelVA and CLAP tensor/model boundaries."""
import math
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tests.sam_audio_fakes import Tensor as AudioTensor


class Tensor(AudioTensor):
    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis))

    def expand(self, *shape):
        return Tensor(np.broadcast_to(self.data, shape))


@dataclass
class SequenceConfig:
    duration: float = 8
    sampling_rate: int = 44100

    @property
    def latent_seq_len(self):
        return math.ceil(self.duration * 44100 / 1024)

    @property
    def clip_seq_len(self):
        return int(self.duration * 8)

    @property
    def sync_seq_len(self):
        return int(((self.duration * 25 - 16) // 8 + 1) * 8)


def runtime(duration=10):
    encoder, generator, features = Mock(), Mock(), Mock()
    for model in (encoder, generator, features):
        model.to.return_value = model
        model.eval.return_value = model
    events = []
    encoder.load_weights.side_effect = lambda state: events.append(('base', state))
    encoder.load_state_dict.side_effect = lambda state, strict: events.append(('sync', state, strict))
    hub = SimpleNamespace(hf_hub_download=Mock(side_effect=lambda repo_id, filename, local_files_only: '/cache/' + filename))
    torch = SimpleNamespace(
        Tensor=Tensor, bfloat16='bf16', inference_mode=nullcontext,
        load=Mock(side_effect=lambda path, **kwargs: {'state_dict': path} if path.endswith('best.pt') else path),
        Generator=Mock(return_value=SimpleNamespace(manual_seed=Mock())),
        cat=lambda tensors, dim=0: Tensor(np.concatenate([t.data for t in tensors], axis=dim)),
        serialization=SimpleNamespace(safe_globals=Mock(side_effect=lambda globals_: nullcontext())),
    )
    evaluation = SimpleNamespace(
        all_model_cfg={'large_44k': SimpleNamespace(model_name='large_44k', mode='44k', seq_cfg=SequenceConfig())},
        load_video=Mock(return_value=SimpleNamespace(sync_frames=Tensor(np.arange(round(duration * 25))[:, None, None, None]), duration_sec=duration)),
        generate=Mock(return_value=Tensor(np.ones((1, 1, 44100), dtype=np.float32))),
    )
    clap = Mock()
    clap.get_audio_embedding_from_data.side_effect = lambda x: np.array([[1., 0.]])
    clap.get_text_embedding.side_effect = lambda text: np.array([[0.8, 0.6], [0., 1.]])[:len(text)]
    modules = {
        'torch': torch,
        'huggingface_hub': hub,
        'selva.utils.eval_utils': evaluation,
        'selva.model.networks_video_enc': SimpleNamespace(get_my_textsynch=Mock(return_value=encoder)),
        'selva.model.networks_generator': SimpleNamespace(get_my_mmaudio=Mock(return_value=generator)),
        'selva.model.flow_matching': SimpleNamespace(FlowMatching=Mock()),
        'selva.model.utils.features_utils': SimpleNamespace(FeaturesUtils=Mock(return_value=features)),
        'laion_clap': SimpleNamespace(CLAP_Module=Mock(return_value=clap)),
    }
    return SimpleNamespace(modules=modules, torch=torch, hub=hub, encoder=encoder, generator=generator, evaluation=evaluation, clap=clap, events=events)
