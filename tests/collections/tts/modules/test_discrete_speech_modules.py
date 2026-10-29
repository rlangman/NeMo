# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from nemo.collections.tts.modules.discrete_speech_modules import DurationDecoder, TextEncoder
from nemo.collections.tts.modules.transformer_2501 import Transformer


def _transformer(is_causal=False):
    return Transformer(n_layers=1, d_model=8, d_ffn=16, sa_n_heads=2, kernel_size=1, is_causal=is_causal)


def _duration_decoder():
    return DurationDecoder(
        parallel_transformer=_transformer(),
        duration_transformer=_transformer(is_causal=True),
        input_dim=8,
        d_context=4,
        d_model=8,
        num_duration=5,
    ).eval()


@pytest.mark.unit
def test_text_encoder_without_context():
    encoder = TextEncoder(
        transformer=_transformer(),
        n_embed=10,
        d_model=8,
        padding_idx=0,
        down_sample_rate=1,
        bos_id=None,
        eos_id=None,
        space_id=1,
        space_dur=1,
    ).eval()
    text = torch.tensor([[1, 2, 3], [1, 2, 0]])
    text_len = torch.tensor([3, 2])

    encoded, lengths, durations = encoder(text=text, text_len=text_len)

    assert encoded.shape == (2, 3, 8)
    torch.testing.assert_close(lengths, text_len)
    torch.testing.assert_close(durations, (text != 0).long())
    assert torch.count_nonzero(encoded[1, 2]) == 0


@pytest.mark.unit
def test_duration_decoder_context_gradients():
    torch.manual_seed(0)
    decoder = _duration_decoder()
    inputs = torch.randn(2, 3, 8)
    original_inputs = inputs.clone()
    context = torch.randn(2, 4, requires_grad=True)

    _, logits, _, parallel_logits = decoder(
        inputs=inputs,
        dur_len=torch.tensor([3, 2]),
        dur_indices=torch.tensor([[1, 2, 3], [2, 1, 0]]),
        speaking_rate=torch.zeros(2),
        context_emb=context,
    )

    for output in (logits, parallel_logits):
        gradient = torch.autograd.grad(output.square().sum(), context, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        assert torch.count_nonzero(output[1, :, 2]) == 0
    torch.testing.assert_close(inputs, original_inputs)


@pytest.mark.unit
@pytest.mark.parametrize("frames_per_iter", [1, 2])
def test_duration_decoder_infer_context(frames_per_iter):
    decoder = _duration_decoder()
    inputs = torch.randn(2, 3, 8)
    original_inputs = inputs.clone()
    context = torch.randn(2, 4)
    projected_inputs = []
    hook = decoder.input_layer.register_forward_pre_hook(
        lambda module, args: projected_inputs.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            indices = decoder.infer(
                inputs=inputs,
                dur_len=torch.tensor([3, 2]),
                speaking_rate=torch.zeros(2),
                context_emb=context,
                frames_per_iter=frames_per_iter,
            )
    finally:
        hook.remove()

    expected = inputs + decoder.context_cond_layer(context).unsqueeze(1)
    torch.testing.assert_close(projected_inputs[0][:, :3], expected)
    torch.testing.assert_close(inputs, original_inputs)
    assert indices.shape == (2, 3)
    assert indices[1, 2] == 0
