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

from types import SimpleNamespace
from unittest.mock import Mock, patch

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


@pytest.mark.unit
@pytest.mark.parametrize("frames_per_iter", [1, 2, 3])
def test_duration_decoder_infer_duration_prefix(frames_per_iter):
    decoder = _duration_decoder()
    prefix = torch.tensor([[4, 2, 3], [2, 0, 0]])
    prefix_len = torch.tensor([3, 1])
    with patch.object(decoder, '_forward_duration', wraps=decoder._forward_duration) as autoregressive:
        with torch.no_grad():
            indices = decoder.infer(
                inputs=torch.randn(2, 5, 8),
                dur_len=torch.tensor([5, 4]),
                speaking_rate=torch.zeros(2),
                context_emb=torch.randn(2, 4),
                frames_per_iter=frames_per_iter,
                context_dur_indices=prefix,
                context_dur_len=prefix_len,
                silence_pad_start=2,
                silence_pad_end=3,
            )

    torch.testing.assert_close(indices[0, :3], prefix[0])
    torch.testing.assert_close(indices[1, :1], prefix[1, :1])
    assert indices[0, 3] == indices[1, 1] == 1
    assert indices[0, 4] == indices[1, 3] == 2
    assert indices[1, 4] == 0
    for call in autoregressive.call_args_list:
        supplied = call.kwargs['dur_indices']
        mask = call.kwargs['cond_mask']
        for batch_idx, length in enumerate(prefix_len):
            known = min(int(length), supplied.shape[1])
            torch.testing.assert_close(supplied[batch_idx, :known], prefix[batch_idx, :known])
            for position in range(1, min(int(length) + 1, supplied.shape[1])):
                assert not mask[batch_idx, position]


@pytest.mark.unit
@pytest.mark.parametrize("predict_speaking_rate", [False, True])
def test_discrete_speech_infer_duration_prefix(predict_speaking_rate):
    from nemo.collections.tts.models.discrete_speech import DiscreteSpeechModel

    text = torch.tensor([[3, 4, 99], [5, 99, 99]])
    text_len = torch.tensor([2, 1])
    context_text = torch.tensor([[1, 2], [6, 99]])
    context_text_len = torch.tensor([2, 1])
    context_tokens = torch.ones(2, 1, 7, dtype=torch.long)
    context_len = torch.tensor([7, 4])
    context_codes = torch.randn(2, 4, 7)
    context_emb = torch.randn(2, 4)
    combined_text = torch.tensor([[1, 2, 3, 4], [6, 5, 99, 99]])
    combined_len = torch.tensor([4, 2])
    context_durs = torch.tensor([[3.0, 4.0], [4.0, 0.0]])
    duration_indices = torch.tensor([[2, 3, 1, 2], [3, 3, 0, 0]])
    audio_tokens = torch.ones(2, 1, 5, dtype=torch.long)
    model = SimpleNamespace(
        text_pad_token=99,
        max_token_duration=20,
        get_context=Mock(return_value=context_emb),
        vector_quantizer=SimpleNamespace(decode=Mock(return_value=context_codes)),
        aligner=Mock(return_value=(context_durs, context_text_len, None, None, None)),
        text_encoder=Mock(return_value=(combined_text.float().unsqueeze(-1), combined_len, None)),
        duration_decoder=SimpleNamespace(infer=Mock(return_value=duration_indices)),
        decoder=SimpleNamespace(infer=Mock(return_value=audio_tokens)),
        speaking_rate_predictor=Mock(return_value=(torch.zeros(2, dtype=torch.long), None)),
        speaking_rate_quantizer=SimpleNamespace(get_codes=Mock(return_value=torch.zeros(2))),
    )
    model.duration_to_index = lambda **kwargs: DiscreteSpeechModel.duration_to_index.__wrapped__(model, **kwargs)
    model.index_to_duration = lambda **kwargs: DiscreteSpeechModel.index_to_duration(model, **kwargs)

    tokens, lengths = DiscreteSpeechModel.infer.__wrapped__(
        model,
        text=text,
        text_len=text_len,
        context_text=context_text,
        context_text_len=context_text_len,
        context_tokens=context_tokens,
        context_len=context_len,
        speaking_rate=None if predict_speaking_rate else torch.zeros(2),
    )

    encoded = model.text_encoder.call_args.kwargs
    torch.testing.assert_close(encoded['text'], combined_text)
    torch.testing.assert_close(encoded['text_len'], combined_len)
    aligned = model.aligner.call_args.kwargs
    torch.testing.assert_close(aligned['text'], context_text)
    torch.testing.assert_close(aligned['text_len'], context_text_len)
    torch.testing.assert_close(aligned['audio_codes'], context_codes)
    torch.testing.assert_close(aligned['audio_len'], context_len)
    torch.testing.assert_close(model.vector_quantizer.decode.call_args.kwargs['input_len'], context_len)
    duration = model.duration_decoder.infer.call_args.kwargs
    torch.testing.assert_close(duration['context_dur_indices'], torch.tensor([[2, 3], [3, 0]], dtype=torch.int))
    torch.testing.assert_close(duration['context_dur_len'], context_text_len)
    generated = model.decoder.infer.call_args.kwargs
    expected = torch.tensor([[[3.0], [3.0], [4.0], [4.0], [4.0]], [[5.0], [5.0], [5.0], [5.0], [0.0]]])
    torch.testing.assert_close(generated['inputs'], expected)
    torch.testing.assert_close(lengths, torch.tensor([5, 4]))
    torch.testing.assert_close(tokens, audio_tokens)
