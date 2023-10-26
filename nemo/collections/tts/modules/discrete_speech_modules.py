# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from einops import rearrange
import math
import torch

from nemo.collections.tts.modules.acoustic_model_modules import Conv1d, TextDownSampling, sample_tokens
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    MaskType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType


class SpeakingRateQuantizer(NeuralModule):

    def __init__(self, num_bins, min_value, max_value):
        super().__init__()
        self.num_bins = num_bins
        self.max_bin = num_bins - 1
        self.shift = (min_value + max_value) / 2
        self.scale = max_value - self.shift

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(tuple('B'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return {
            "codes": NeuralType(tuple('B'), EncodedRepresentation()),
            "indices": NeuralType(tuple('B'), TokenIndex()),
        }

    @typecheck()
    def forward(self, inputs):
        scaled = (inputs - self.shift) / self.scale
        # [-1, 1]
        scaled = torch.clamp(scaled, min=-1.0, max=1.0)
        # [0, 1]
        shifted = (scaled + 1.0) / 2.0
        # [0, num_bins]
        indices = torch.round(shifted * self.max_bin)
        indices = torch.clamp(indices, min=0, max=self.max_bin).int()
        codes = self.get_codes(indices)

        return codes, indices

    def get_codes(self, indices):
        codes = indices.float() / self.max_bin
        codes = 2.0 * codes - 1.0
        return codes


class SpeakingRatePredictor(NeuralModule):

    def __init__(self, num_speaking_rate, context_dim):
        super(SpeakingRatePredictor, self).__init__()
        self.hidden_layer = torch.nn.Linear(context_dim, context_dim)
        self.speaking_rate_layer = torch.nn.Linear(context_dim, num_speaking_rate)

    @property
    def input_types(self):
        return {"context_emb": NeuralType(('B', 'D'), EncodedRepresentation())}

    @property
    def output_types(self):
        return {
            "speaking_rate_indices_pred": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_logits": NeuralType(('B', 'C'), LogitsType()),
        }

    @typecheck()
    def forward(self, context_emb):
        out = self.hidden_layer(context_emb)
        # [B, num_sr]
        speaking_rate_logits = self.speaking_rate_layer(out)
        # [B]
        speaking_rate_indices_pred = speaking_rate_logits.max(dim=1).indices
        return speaking_rate_indices_pred, speaking_rate_logits


class ContextEncoder(NeuralModule):

    def __init__(
        self,
        input_dim,
        d_model,
        transformer,
        rnn_layers,
        rnn_dim,
    ):
        super(ContextEncoder, self).__init__()
        self.pre_conv = Conv1d(in_channels=input_dim, out_channels=d_model)
        self.transformer = transformer
        self.rnn = torch.nn.LSTM(input_size=d_model, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True)
        self.emb_layer = torch.nn.Linear(in_features=rnn_dim, out_features=d_model)

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)
        context = self.pre_conv(inputs=audio_codes, mask=mask)
        context = rearrange(context, 'B D T -> B T D')
        context = self.transformer(x=context, x_mask=mask)['output']

        out = torch.nn.utils.rnn.pack_padded_sequence(
            context, audio_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(out)
        out, padded_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # [B, D]
        out = out[torch.arange(len(padded_lens)), (padded_lens - 1), :]
        context_emb = self.emb_layer(out)

        context = rearrange(context, 'B T D -> B D T')

        return context_emb, context


class TextEncoder(NeuralModule):
    def __init__(
        self,
        transformer,
        n_embed,
        d_model,
        d_context,
        padding_idx,
        down_sample_rate,
        bos_id,
        eos_id,
        space_id,
        space_dur,
    ):
        super(TextEncoder, self).__init__()
        self.d_model = d_model

        self.word_emb = torch.nn.Embedding(n_embed, d_model, padding_idx=padding_idx)
        self.context_cond_layer = torch.nn.Linear(d_context, self.d_model)
        self.transformer = transformer
        if down_sample_rate and down_sample_rate > 1:
            self.downsample_layer = TextDownSampling(
                input_dim=d_model,
                down_sample_rate=down_sample_rate,
                bos_id=bos_id,
                eos_id=eos_id,
                space_id=space_id,
                space_dur=space_dur,
            )
        else:
            self.downsample_layer = None

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), EncodedRepresentation()),
            "text_lens": NeuralType(('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "out_lens": NeuralType(tuple('B'), LengthsType()),
            "text_durs": NeuralType(('B', 'T'), LengthsType()),
        }

    @typecheck()
    def forward(self, text, text_lens, context_emb):
        text_mask = get_mask_from_lengths(text_lens)
        text_emb = self.word_emb(text)

        out = self.transformer(x=text_emb, x_mask=text_mask)['output']

        if self.downsample_layer is not None:
            out = rearrange(out, 'B T D -> B D T')
            out, out_lens, text_durs = self.downsample_layer(text=text, text_emb=out, text_lens=text_lens)
            out = rearrange(out, 'B D T -> B T D')
        else:
            out_lens = text_lens
            text_durs = torch.ones_like(text)
            text_durs = text_durs * text_mask

        out_mask = get_mask_from_lengths(out_lens)

        context_emb = rearrange(context_emb, 'B D -> B 1 D')
        context_res = self.context_cond_layer(context_emb)
        out = out + context_res
        out = out * rearrange(out_mask, 'B T -> B T 1')

        return out, out_lens, text_durs


class DurationDecoder(NeuralModule):

    def __init__(self, transformer, d_model, num_duration):
        super(DurationDecoder, self).__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.num_duration = num_duration

        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))

        self.dur_cond_layer = torch.nn.Linear(1, self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.duration_layer = torch.nn.Linear(self.d_model, self.num_duration)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
            "duration_maskin": NeuralType(('B', 'T_text'), MaskType()),
            "temperature": NeuralType((), FloatType(), optional=True),
            "topk": NeuralType((), IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'C', 'T_text'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, dur_indices, text_mask, duration_maskin=None, temperature=None, topk=None):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        dur_indices_shifted = dur_indices[:, :-1]
        dur_indices_shifted = torch.nn.functional.pad(dur_indices_shifted, pad=(1, 0))

        log_dur = torch.log(dur_indices_shifted + 1.0).detach()
        log_dur = rearrange(log_dur, 'B T -> B T 1')
        dur_res = self.dur_cond_layer(log_dur)

        if duration_maskin is not None:
            duration_maskin_3d = rearrange(duration_maskin, 'B T -> B T 1')
            dur_res = torch.where(duration_maskin_3d, dur_res, self.mask_emb)

        dec_input = inputs + dur_res
        dec_input = dec_input * text_mask_3d

        # [B, T, D]
        dec_input = dec_input * rearrange(text_mask, 'B T -> B T 1')
        dec_out = self.transformer(x=dec_input, x_mask=text_mask)['output']

        # [B, T, num_codes]
        dec_out = self.layer_norm(dec_out)
        dur_logits = self.duration_layer(dec_out)
        dur_logits = dur_logits * text_mask_3d

        # [B, T]
        if temperature is None:
            dur_indices_pred = dur_logits.max(dim=2).indices
        else:
            dur_indices_pred = sample_tokens(logits=dur_logits, temperature=temperature, topk=topk)

        dur_indices_pred = dur_indices_pred * text_mask
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        return dur_indices_pred, dur_logits

    def infer(
        self,
        inputs,
        text_lens,
        frames_per_iter,
        temperature=None,
        topk=None,
        silence_pad_start=None,
        silence_pad_end=None,
    ):
        self.transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)

        batch_size = inputs.shape[0]
        # [B, T]
        dur_mask = get_mask_from_lengths(text_lens)

        dur_lens_padded = torch.ceil(text_lens / frames_per_iter).int() * frames_per_iter
        max_len = text_lens.max()
        max_len_padded = dur_lens_padded.max()
        inputs = torch.nn.functional.pad(inputs, (0, 0, 0, max_len_padded - max_len))

        # [B, T]
        dur_indices_shape = [batch_size, max_len_padded]
        dur_indices = torch.zeros(dur_indices_shape, dtype=torch.int, device=inputs.device)

        for i in range(0, max_len_padded, frames_per_iter):
            inputs_i = inputs[:, : i + frames_per_iter, :]
            dur_indices_i = dur_indices[:, : i + frames_per_iter]
            dur_mask_i = dur_mask[:, : i + frames_per_iter]

            dur_maskin_i = dur_mask_i.clone()
            for j in range(1, frames_per_iter):
                dur_maskin_i[:, i + j] = False

            # [B, C, T], [B, C, W, T]
            dur_indices_i, _ = self.forward(
                inputs=inputs_i,
                text_mask=dur_mask_i,
                dur_indices=dur_indices_i,
                duration_maskin=dur_maskin_i,
                temperature=temperature,
                topk=topk,
            )

            for j in range(frames_per_iter):
                dur_indices[:, i + j] = dur_indices_i[:, i + j]

            if i == 0 and silence_pad_start:
                dur_indices[:, 0] = silence_pad_start - 1

        dur_indices = dur_indices[:, :max_len]

        if silence_pad_end:
            for i in range(dur_indices.shape[0]):
                last_i = text_lens[i] - 1
                dur_indices[i, last_i] = silence_pad_end - 1

        dur_indices = dur_indices * dur_mask

        self.transformer.reset_cache(use_cache=False)

        return dur_indices

    def infer_diffusion(
        self,
        inputs,
        text_lens,
        num_iters,
        temperature=None,
        topk=None,
        silence_pad_start=None,
        silence_pad_end=None,
    ):
        # [B, T]
        text_mask = get_mask_from_lengths(text_lens)

        num_tokens = inputs.shape[1]
        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        duration_maskin = torch.zeros_like(text_mask, dtype=torch.bool)
        duration_maskin[:, 0] = True
        dur_indices = torch.zeros_like(text_mask, dtype=torch.int)

        if silence_pad_start:
            for i in range(dur_indices.shape[0]):
                dur_indices[i, 0] = silence_pad_start - 1

        if silence_pad_end:
            for i in range(dur_indices.shape[0]):
                last_i = text_lens[i] - 1
                dur_indices[i, last_i] = silence_pad_end - 1
                duration_maskin[i, last_i] = True

        for i in range(num_iters):
            dur_indices_i, dur_logits = self(
                inputs=inputs,
                dur_indices=dur_indices,
                text_mask=text_mask,
                duration_maskin=duration_maskin,
                temperature=temperature,
                topk=topk,
            )

            top_i = torch.clamp_max(index_shift + 1, max=num_tokens - 1)

            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)
            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(text_mask, maskin_i, False)

            dur_indices = torch.where(maskin_i, dur_indices_i, dur_indices)

            next_i = torch.clamp_max(index_shift + i + 1, max=num_tokens - 1)
            # [B, T // num_iters, T]
            next_one_hot = torch.nn.functional.one_hot(next_i, num_classes=num_tokens)
            # [B, T]
            next_maskin = next_one_hot.sum(dim=1).bool()
            next_maskin = torch.where(text_mask, next_maskin, False)

            duration_maskin = torch.logical_or(duration_maskin, next_maskin)

        dur_indices = torch.where(duration_maskin, dur_indices, dur_indices_i)

        return dur_indices


class DurationDiffusionDecoder(NeuralModule):

    def __init__(self, transformer, d_model, num_duration, dropout_rate=0.1):
        super(DurationDiffusionDecoder, self).__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.num_duration = num_duration

        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))

        self.dur_cond_layer = torch.nn.Linear(1, self.d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.duration_layer = torch.nn.Linear(self.d_model, self.num_duration)
        self.layer_norm_parallel = torch.nn.LayerNorm(self.d_model)
        self.duration_layer_prallel = torch.nn.Linear(self.d_model, self.num_duration)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
            "duration_maskin": NeuralType(('B', 'T_text'), MaskType()),
            "temperature": NeuralType((), FloatType(), optional=True),
            "topk": NeuralType((), IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'C', 'T_text'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, dur_indices, text_mask, duration_maskin, temperature=None, topk=None):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        log_dur = torch.log(dur_indices + 1.0).detach()
        log_dur = rearrange(log_dur, 'B T -> B T 1')
        dur_res = self.dur_cond_layer(log_dur)
        dur_res = self.dropout(dur_res)
        dur_res = dur_res * rearrange(duration_maskin, 'B T -> B T 1')

        masked_mask = ~duration_maskin * text_mask
        mask_res = self.mask_emb * rearrange(masked_mask, 'B T -> B T 1')

        dec_input = inputs + dur_res + mask_res

        # [B, T, D]
        dec_input = dec_input * rearrange(text_mask, 'B T -> B T 1')
        dec_out = self.transformer(x=dec_input, x_mask=text_mask)['output']

        # [B, T, num_codes]
        dec_out = self.layer_norm(dec_out)
        dur_logits = self.duration_layer(dec_out)
        dur_logits = dur_logits * text_mask_3d

        # [B, T]
        if temperature is None:
            dur_indices_pred = dur_logits.max(dim=2).indices
        else:
            dur_indices_pred = sample_tokens(logits=dur_logits, temperature=temperature, topk=topk)

        dur_indices_pred = dur_indices_pred * text_mask
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        return dur_indices_pred, dur_logits

    def forward_parallel(self, inputs, text_mask, temperature=None, topk=None):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        # [B, T, num_codes]
        dec_out = self.layer_norm_parallel(inputs)
        dur_logits = self.duration_layer_prallel(dec_out)
        dur_logits = dur_logits * text_mask_3d

        # [B, T]
        if temperature is None:
            dur_indices_pred = dur_logits.max(dim=2).indices
        else:
            dur_indices_pred = sample_tokens(logits=dur_logits, temperature=temperature, topk=topk)

        dur_indices_pred = dur_indices_pred * text_mask
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        return dur_indices_pred, dur_logits


class DurationEncoder(NeuralModule):
    def __init__(self, input_dim, d_model, transformer):
        super(DurationEncoder, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, d_model)
        self.speaking_rate_cond_layer = torch.nn.Linear(1, d_model)
        self.transformer = transformer


    @property
    def input_types(self):
        return {
            "text_enc": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),

        }

    def forward(self, text_enc, text_mask, speaking_rate, context, context_mask):
        speaking_rate = rearrange(speaking_rate, 'B -> B 1 1')
        # [B, T, hidden_dim]
        sr_res = self.speaking_rate_cond_layer(speaking_rate)

        dur_enc = self.input_layer(text_enc)
        dur_enc = dur_enc + sr_res
        dur_enc = dur_enc * rearrange(text_mask, 'B T -> B T 1')

        dur_enc = self.transformer(
            x=dur_enc, x_mask=text_mask, cond=context, cond_mask=context_mask
        )['output']

        return dur_enc


class AudioEncoder(NeuralModule):
    def __init__(self, input_dim, d_model, transformer):
        super(AudioEncoder, self).__init__()

        self.input_layer = torch.nn.Linear(input_dim, d_model)
        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
            "encoder_mask": NeuralType(('B', 'T_audio'), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, audio_mask, context, context_mask, encoder_mask=None):
        audio_enc = self.input_layer(inputs)

        if encoder_mask is not None:
            encoder_mask_3d = rearrange(encoder_mask, 'B T -> B T 1')
            audio_enc = torch.where(encoder_mask_3d, self.mask_emb, audio_enc)

        audio_enc = self.transformer(
            x=audio_enc, x_mask=audio_mask, cond=context, cond_mask=context_mask
        )['output']

        return audio_enc