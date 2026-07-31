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
        output_dim,
        d_model,
        transformer,
    ):
        super(ContextEncoder, self).__init__()
        self.pre_conv1 = Conv1d(in_channels=input_dim, out_channels=d_model, activation="gelu")
        self.pre_conv2 = Conv1d(in_channels=d_model, out_channels=d_model)
        self.transformer = transformer
        self.input_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))
        self.emb_layer = torch.nn.Linear(in_features=d_model, out_features=output_dim)

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
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        batch_size = audio_codes.size(0)
        mask = get_mask_from_lengths(audio_lens)

        context = self.pre_conv1(inputs=audio_codes, mask=mask)
        context = self.pre_conv2(inputs=context, mask=mask)

        context = rearrange(context, 'B D T -> B T D')
        context_input_emb = self.input_emb.tile([batch_size, 1, 1])

        context_input_len = audio_lens + 1
        context_mask = get_mask_from_lengths(context_input_len)
        context = torch.concat([context_input_emb, context], dim=1)
        context = self.transformer(x=context, x_mask=context_mask)['output']

        context_emb = context[:, 0, :]
        context_emb = self.emb_layer(context_emb)

        return context_emb


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
        self.duration_layer = torch.nn.Linear(self.d_model, self.num_duration)
        self.layer_norm_parallel = torch.nn.LayerNorm(self.d_model)
        self.duration_layer_parallel = torch.nn.Linear(self.d_model, self.num_duration)

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

    def forward_parallel(self, inputs, text_mask, temperature=None, topk=None):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        # [B, T, num_codes]
        out = self.layer_norm_parallel(inputs)
        dur_logits = self.duration_layer_parallel(out)
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
        infer_weight=1.0,
        temperature=None,
        topk=None,
        silence_pad_start=None,
        silence_pad_end=None,
    ):
        ar_weight = infer_weight / (1.0 + infer_weight)
        parallel_weight = 1.0 / (1.0 + infer_weight)
        batch_size = inputs.shape[0]
        # [B, T]
        dur_mask = get_mask_from_lengths(text_lens, pad_to_factor=frames_per_iter)

        dur_lens_padded = torch.ceil(text_lens / frames_per_iter).int() * frames_per_iter
        max_len = text_lens.max()
        max_len_padded = dur_lens_padded.max()
        inputs = torch.nn.functional.pad(inputs, (0, 0, 0, max_len_padded - max_len))

        # [B, T]
        dur_indices_shape = [batch_size, max_len_padded]
        dur_indices = torch.zeros(dur_indices_shape, dtype=torch.int, device=inputs.device)

        _, logits_parallel = self.forward_parallel(
            inputs=inputs,
            text_mask=dur_mask,
            temperature=temperature,
            topk=topk,
        )
        logits_parallel = rearrange(logits_parallel, 'B C T -> B T C')

        self.transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)


        for i in range(0, max_len_padded, frames_per_iter):
            inputs_i = inputs[:, : i + frames_per_iter, :]
            dur_indices_i = dur_indices[:, : i + frames_per_iter]
            dur_mask_i = dur_mask[:, : i + frames_per_iter]

            dur_maskin_i = dur_mask_i.clone()
            for j in range(1, frames_per_iter):
                dur_maskin_i[:, i + j] = False

            # [B, C, T], [B, C, W, T]
            _, logits_i = self.forward(
                inputs=inputs_i,
                text_mask=dur_mask_i,
                dur_indices=dur_indices_i,
                duration_maskin=dur_maskin_i,
                temperature=temperature,
                topk=topk,
            )
            logits_i = rearrange(logits_i, 'B C T -> B T C')
            logits = (parallel_weight * logits_parallel[:, :i + frames_per_iter]) + (ar_weight * logits_i)
            dur_indices_i = logits.max(dim=2).indices

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


class DurationEncoder(NeuralModule):
    def __init__(self, input_dim, d_model, transformer):
        super(DurationEncoder, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, d_model)
        self.speaking_rate_cond_layer = torch.nn.Linear(1, d_model)
        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))
        self.transformer = transformer


    @property
    def input_types(self):
        return {
            "text_enc": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
            "encoder_mask": NeuralType(('B', 'T_audio'), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),

        }

    def forward(self, text_enc, text_mask, speaking_rate, encoder_mask=None):
        speaking_rate = rearrange(speaking_rate, 'B -> B 1 1')
        # [B, T, hidden_dim]
        sr_res = self.speaking_rate_cond_layer(speaking_rate)

        dur_enc = self.input_layer(text_enc)
        dur_enc = dur_enc + sr_res
        dur_enc = dur_enc * rearrange(text_mask, 'B T -> B T 1')

        if encoder_mask is not None:
            encoder_mask_3d = rearrange(encoder_mask, 'B T -> B T 1')
            dur_enc = torch.where(encoder_mask_3d, dur_enc, self.mask_emb)

        dur_enc = self.transformer(x=dur_enc, x_mask=text_mask)['output']

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
            "encoder_mask": NeuralType(('B', 'T_audio'), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, audio_mask, encoder_mask=None):
        audio_enc = self.input_layer(inputs)

        if encoder_mask is not None:
            encoder_mask_3d = rearrange(encoder_mask, 'B T -> B T 1')
            audio_enc = torch.where(encoder_mask_3d, audio_enc, self.mask_emb)

        audio_enc = self.transformer(x=audio_enc, x_mask=audio_mask)['output']

        return audio_enc


class DiscreteSpeechDecoder(NeuralModule):

    def __init__(
        self,
        parallel_transformer,
        semantic_transformer,
        acoustic_transformer,
        input_dim,
        d_model,
        num_semantic_codebooks,
        num_acoustic_codebooks,
        codebook_size,
        codebook_dim,
        semantic_dim,
        infill_min=0.25,
        infill_max=1.0,
    ):
        super(DiscreteSpeechDecoder, self).__init__()
        self.num_semantic_codebooks = num_semantic_codebooks
        self.num_acoustic_codebooks = num_acoustic_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.semantic_dim = semantic_dim
        self.num_semantic_logits = self.num_semantic_codebooks * self.codebook_size
        self.num_acoustic_logits = self.num_acoustic_codebooks * self.codebook_size

        self.input_layer = torch.nn.Linear(input_dim, d_model)

        self.parallel_transformer = parallel_transformer
        self.semantic_transformer = semantic_transformer
        self.acoustic_transformer = acoustic_transformer

        self.audio_hidden_layer = torch.nn.Linear(codebook_dim, d_model)
        self.audio_cond_layer = torch.nn.Linear(d_model, d_model)

        self.semantic_hidden_layer = torch.nn.Linear(semantic_dim, d_model)
        self.semantic_cond_layer = torch.nn.Linear(d_model, d_model)

        self.semantic_layer_norm_parallel = torch.nn.LayerNorm(d_model)
        self.semantic_token_layer_parallel = torch.nn.Linear(d_model, self.num_semantic_logits)

        self.semantic_layer_norm = torch.nn.LayerNorm(d_model)
        self.semantic_token_layer = torch.nn.Linear(d_model, self.num_semantic_logits)
        
        self.acoustic_layer_norm = torch.nn.LayerNorm(d_model)
        self.acoustic_token_layer = torch.nn.Linear(d_model, self.num_acoustic_logits)

        self.input_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))
        self.audio_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))
        self.semantic_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))

        self.infill_min = infill_min
        self.infill_max = infill_max
        self.infill_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=2.0)

    def create_infill_mask(self, input_lens):
        batch_size = input_lens.shape[0]
        len_mask = get_mask_from_lengths(input_lens)
        max_len = len_mask.shape[1]

        infill_percent = self.infill_dist.sample(sample_shape=torch.Size([batch_size])).to(input_lens.device)
        infill_percent = self.infill_min + (self.infill_max - self.infill_min) * infill_percent
        infill_len = infill_percent * input_lens.float()
        infill_rank = torch.clamp_min(infill_len - 1, 0).long()
        infill_rank = rearrange(infill_rank, 'B -> B 1')

        # [batch_size, time]
        infill_vals = torch.rand(size=len_mask.shape, device=input_lens.device)
        infill_vals = infill_vals * len_mask
        infill_topk = torch.topk(infill_vals, k=max_len, dim=1, sorted=True).values
        infill_min_val = torch.gather(infill_topk, index=infill_rank, dim=1)
        infill_mask = infill_vals >= infill_min_val

        infill_mask = infill_mask * len_mask

        return infill_mask

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
            "semantic_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return {
            "semantic_tokens_parallel": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits_parallel": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "semantic_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "acoustic_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "acoustic_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, audio_lens, audio_codes, semantic_codes):
        audio_mask = get_mask_from_lengths(audio_lens)
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        hidden_state = self.input_layer(inputs)

        if self.training:
            input_infill_mask = self.create_infill_mask(input_lens=audio_lens)
            input_infill_mask = rearrange(input_infill_mask, 'B T -> B T 1')
            hidden_state = torch.where(input_infill_mask, hidden_state, self.input_mask_emb)

        hidden_state = self.parallel_transformer(x=hidden_state, x_mask=audio_mask)['output']

        semantic_tokens_parallel, semantic_logits_parallel = self._compute_logits(
            inputs=hidden_state,
            audio_mask=audio_mask,
            layer_norm=self.semantic_layer_norm_parallel,
            projection=self.semantic_token_layer_parallel,
            num_codebooks=self.num_semantic_codebooks,
        )

        audio_codes_shifted = audio_codes[:, :-1, :]
        audio_codes_shifted = torch.nn.functional.pad(audio_codes_shifted, pad=(0, 0, 1, 0))

        audio_res = self.audio_hidden_layer(audio_codes_shifted)
        audio_res = self.audio_cond_layer(audio_res)

        if self.training:
            infill_mask = self.create_infill_mask(input_lens=audio_lens)
            infill_mask = rearrange(infill_mask, 'B T -> B T 1')
            audio_res = torch.where(infill_mask, audio_res, self.audio_mask_emb)

        hidden_state = hidden_state + audio_res
        hidden_state = hidden_state * audio_mask_3d

        # [batch_size, audio_len, hidden_dim]
        hidden_state = self.semantic_transformer(x=hidden_state, x_mask=audio_mask)['output']

        semantic_tokens, semantic_logits= self._compute_logits(
            inputs=hidden_state,
            audio_mask=audio_mask,
            layer_norm=self.semantic_layer_norm,
            projection=self.semantic_token_layer,
            num_codebooks=self.num_semantic_codebooks,
        )

        semantic_res = self.semantic_hidden_layer(semantic_codes)
        semantic_res = self.semantic_cond_layer(semantic_res)

        #if self.training:
        #    semantic_infill_mask = self.create_infill_mask(input_lens=audio_lens)
        #    semantic_infill_mask = rearrange(semantic_infill_mask, 'B T -> B T 1')
        #    semantic_res = torch.where(semantic_infill_mask, semantic_res, self.semantic_mask_emb)

        if self.training:
            semantic_res = torch.where(infill_mask, semantic_res, self.semantic_mask_emb)

        hidden_state = hidden_state + semantic_res
        hidden_state = hidden_state * audio_mask_3d

        # [batch_size, audio_len, hidden_dim]
        hidden_state = self.acoustic_transformer(x=hidden_state, x_mask=audio_mask)['output']

        acoustic_tokens, acoustic_logits = self._compute_logits(
            inputs=hidden_state,
            audio_mask=audio_mask,
            layer_norm=self.acoustic_layer_norm,
            projection=self.acoustic_token_layer,
            num_codebooks=self.num_acoustic_codebooks,
        )

        return semantic_tokens_parallel, semantic_logits_parallel, semantic_tokens, semantic_logits, acoustic_tokens, acoustic_logits


    def _compute_logits(self, inputs, audio_mask, layer_norm, projection, num_codebooks):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        # [batch_size, audio_len, num_codebook * codebook_size]
        out = layer_norm(inputs)
        audio_logits = projection(out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], num_codebooks, self.codebook_size)
        audio_logits = torch.reshape(audio_logits, logit_shape)

        # [batch_size, audio_len, num_codebook]
        audio_tokens = audio_logits.max(dim=3).indices
        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits

    def infer(
        self,
        inputs,
        audio_lens,
        frames_per_iter,
        vector_quantizer,
        infer_weight=1.0,
    ):
        ar_weight = infer_weight / (1.0 + infer_weight)
        parallel_weight = 1.0 / (1.0 + infer_weight)
        batch_size = inputs.shape[0]
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens, pad_to_factor=frames_per_iter)

        audio_lens_padded = torch.ceil(audio_lens / frames_per_iter).int() * frames_per_iter
        max_len = audio_lens.max()
        max_len_padded = audio_lens_padded.max()
        inputs = torch.nn.functional.pad(inputs, (0, 0, 0, max_len_padded - max_len))

        # [B, T, C]
        audio_token_shape = [batch_size, max_len_padded, self.num_semantic_codebooks + self.num_acoustic_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [batch_size, max_len_padded, self.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)
        # [B, T, D]
        semantic_code_shape = [batch_size, max_len_padded, self.semantic_dim]
        semantic_codes = torch.zeros(semantic_code_shape, dtype=torch.float, device=inputs.device)

        hidden_state_input = self.input_layer(inputs)
        hidden_state_input = self.parallel_transformer(x=hidden_state_input, x_mask=audio_mask)['output']

        _, logits_parallel = self._compute_logits(
            inputs=hidden_state_input,
            audio_mask=audio_mask,
            layer_norm=self.semantic_layer_norm_parallel,
            projection=self.semantic_token_layer_parallel,
            num_codebooks=self.num_semantic_codebooks,
        )
        logits_parallel = rearrange(logits_parallel, 'B C W T -> B T C W')

        self.semantic_transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)
        self.acoustic_transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)

        for i in range(0, max_len_padded, frames_per_iter):
            audio_codes_i = audio_codes[:, : i + frames_per_iter, :]
            audio_mask_i = audio_mask[:, : i + frames_per_iter]
            hidden_state_i = hidden_state_input[:, : i + frames_per_iter, :]

            audio_maskin_i = audio_mask_i.clone()
            for j in range(1, frames_per_iter):
                audio_maskin_i[:, i + j] = False
            audio_maskin_i = rearrange(audio_maskin_i, 'B T -> B T 1')

            audio_codes_shifted = audio_codes_i[:, :-1, :]
            audio_codes_shifted = torch.nn.functional.pad(audio_codes_shifted, pad=(0, 0, 1, 0))
            audio_res = self.audio_hidden_layer(audio_codes_shifted)
            audio_res = self.audio_cond_layer(audio_res)
            audio_res = torch.where(audio_maskin_i, audio_res, self.audio_mask_emb)

            hidden_state_i = hidden_state_i + audio_res
            hidden_state_i = hidden_state_i * rearrange(audio_mask_i, 'B T -> B T 1')

            # [batch_size, audio_len, hidden_dim]
            hidden_state_i = self.semantic_transformer(x=hidden_state_i, x_mask=audio_mask_i)['output']

            _, logits_i = self._compute_logits(
                inputs=hidden_state_i,
                audio_mask=audio_mask_i,
                layer_norm=self.semantic_layer_norm,
                projection=self.semantic_token_layer,
                num_codebooks=self.num_semantic_codebooks,
            )
            logits_i = rearrange(logits_i, 'B C W T -> B T C W')
            logits = (parallel_weight * logits_parallel[:, :i + frames_per_iter]) + (ar_weight * logits_i)
            semantic_tokens_i = logits.max(dim=3).indices

            semantic_tokens_rearrange_i = rearrange(semantic_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            semantic_codes_pred_i = vector_quantizer.decode(indices=semantic_tokens_rearrange_i, input_len=audio_lens)
            semantic_codes_pred_i = rearrange(semantic_codes_pred_i, 'B D T -> B T D')

            for j in range(frames_per_iter):
                semantic_codes[:, i + j, :] = semantic_codes_pred_i[:, i + j, :]

            semantic_codes_i = semantic_codes[:, : i + frames_per_iter, :]

            semantic_res = self.semantic_hidden_layer(semantic_codes_i)
            semantic_res = self.semantic_cond_layer(semantic_res)
            semantic_res = torch.where(audio_maskin_i, semantic_res, self.semantic_mask_emb)

            hidden_state_i = hidden_state_i + semantic_res
            hidden_state_i = hidden_state_i * rearrange(audio_mask_i, 'B T -> B T 1')

            # [batch_size, audio_len, hidden_dim]
            hidden_state_i = self.acoustic_transformer(x=hidden_state_i, x_mask=audio_mask_i)['output']

            acoustic_tokens_i, _ = self._compute_logits(
                inputs=hidden_state_i,
                audio_mask=audio_mask_i,
                layer_norm=self.acoustic_layer_norm,
                projection=self.acoustic_token_layer,
                num_codebooks=self.num_acoustic_codebooks,
            )
            acoustic_tokens_i = rearrange(acoustic_tokens_i, 'B C T -> B T C')
            acoustic_tokens_rearrange_i = rearrange(acoustic_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            acoustic_codes_pred_i = vector_quantizer.decode(indices=acoustic_tokens_rearrange_i, input_len=audio_lens)
            acoustic_codes_pred_i = rearrange(acoustic_codes_pred_i, 'B D T -> B T D')


            for j in range(frames_per_iter):
                audio_codes_pred_i = torch.concat([semantic_codes_pred_i, acoustic_codes_pred_i], dim=2)
                audio_tokens_i = torch.concat([semantic_tokens_i, acoustic_tokens_i], dim=2)
                audio_codes[:, i + j, :] = audio_codes_pred_i[:, i + j, :]
                audio_tokens[:, i + j, :] = audio_tokens_i[:, i + j, :]

        audio_tokens = audio_tokens[:, :max_len, :]
        audio_mask_unpadded = get_mask_from_lengths(audio_lens)
        audio_tokens = audio_tokens * audio_mask_unpadded.unsqueeze(2)
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        self.semantic_transformer.reset_cache(use_cache=False)
        self.acoustic_transformer.reset_cache(use_cache=False)

        return audio_tokens
