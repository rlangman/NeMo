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
import torch

from nemo.collections.tts.modules.transformer_2501 import Transformer
from nemo.collections.tts.parts.utils.helpers import binarize_attention_parallel, get_mask_from_lengths, regulate_len
from nemo.collections.tts.parts.utils.tts_dataset_utils import beta_binomial_prior_distribution_torch
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    FloatType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    ProbsType,
    TokenDurationType,
    TokenIndex,
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType


def create_feature_mask(input_len, dist, mask_min, mask_max):
    batch_size = input_len.shape[0]
    len_mask = get_mask_from_lengths(input_len)
    max_len = len_mask.shape[1]

    mask_percent = dist.sample(sample_shape=torch.Size([batch_size])).to(input_len.device)
    mask_percent = mask_min + (mask_max - mask_min) * mask_percent
    mask_len = mask_percent * input_len.float()
    mask_rank = torch.clamp_min(mask_len - 1, 0).long()
    mask_rank = rearrange(mask_rank, 'B -> B 1')

    # [batch_size, time]
    mask_vals = torch.rand(size=len_mask.shape, device=input_len.device)
    mask_vals = mask_vals * len_mask
    mask_topk = torch.topk(mask_vals, k=max_len, dim=1, sorted=True).values
    mask_min_val = torch.gather(mask_topk, index=mask_rank, dim=1)
    mask = mask_vals >= mask_min_val

    mask = mask * len_mask

    return mask


def sample_tokens(logits, topk, temperature):
    batch_shape = logits.shape[:-1]
    # [B, codebook_size]
    logits = logits.reshape(batch_shape.numel(), -1)
    # [B, k]
    logits_topk = torch.topk(logits, topk, dim=1)[0]
    # [B, 1]
    min_logits = logits_topk[:, -1:]
    # [B, codebook_size]
    indices_to_remove = logits < min_logits
    # [B, codebook_size]
    logits_rescored = logits.clone()
    logits_rescored = logits_rescored / temperature
    logits_rescored[indices_to_remove] = float('-inf')

    probs = torch.softmax(logits_rescored, dim=1)
    # [(B * T * num_codebook), 1]
    tokens = torch.multinomial(input=probs, num_samples=1)
    tokens = tokens.reshape(batch_shape)
    return tokens


class Conv1d(NeuralModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if activation is None:
            self.activation = None
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    @typecheck()
    def forward(self, inputs, mask):
        out = self.conv(inputs)
        if self.activation:
            out = self.activation(out)
        out = out * rearrange(mask, 'B T -> B 1 T')
        return out


class FeatureMasking(NeuralModule):
    def __init__(self, mask_min: float, mask_max: float, mask_alpha: float = 2.0, mask_beta: float = 1.0):
        super().__init__()
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.mask_dist = torch.distributions.beta.Beta(concentration1=mask_alpha, concentration0=mask_beta)

    def forward(self, inputs, mask):
        mask = rearrange(mask, 'B T -> B T 1')
        out = torch.where(mask, torch.zeros_like(inputs), inputs)
        return out

    def create_mask(self, input_len):
        mask = create_feature_mask(
            input_len=input_len, dist=self.mask_dist, mask_min=self.mask_min, mask_max=self.mask_max
        )
        return mask

    def apply_dropout(self, inputs, input_len):
        mask = self.create_mask(input_len=input_len)
        out = self.forward(inputs=inputs, mask=mask)
        return out


class Aligner(NeuralModule):

    def __init__(
        self,
        alignment_encoder,
        num_text_emb,
        text_emb_dim,
        prior_scaling_factor=0.2,
        down_sample_rate=None,
        space_id=None,
        bos_id=None,
        eos_id=None,
        space_dur=None,
    ):
        super().__init__()
        self.alignment_encoder = alignment_encoder
        self.text_emb = torch.nn.Embedding(num_text_emb, text_emb_dim)
        self.prior_scaling_factor = prior_scaling_factor

        if down_sample_rate and down_sample_rate > 1:
            self.downsample_layer = TextDownSampling(
                input_dim=text_emb_dim,
                down_sample_rate=down_sample_rate,
                bos_id=bos_id,
                eos_id=eos_id,
                space_id=space_id,
                space_dur=space_dur,
            )
        else:
            self.downsample_layer = None

    def _create_alignment_prior(self, text_len, text_max_len, audio_len, audio_max_len):
        batch_size = text_len.shape[0]
        prior_batch = torch.zeros([batch_size, audio_max_len, text_max_len], device=text_len.device)
        for i in range(batch_size):
            text_len_i = text_len[i].item()
            audio_len_i = audio_len[i].item()
            prior = beta_binomial_prior_distribution_torch(
                phoneme_count=text_len_i,
                mel_count=audio_len_i,
                scaling_factor=self.prior_scaling_factor,
                device=text_len.device,
            ).to(text_len.device)
            prior_batch[i, :audio_len_i, :text_len_i] = prior

        return prior_batch

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_len": NeuralType(tuple('B'), LengthsType()),
            "audio_codes": NeuralType(('B', 'D', 'T_audio'), EncodedRepresentation()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation(), optional=True),
        },
        output_types={
            "durations": NeuralType(('B', 'T_text'), TokenDurationType()),
            "duration_len": NeuralType(tuple('B'), LengthsType()),
            "attn_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        },
    )
    def forward(self, text, text_len, audio_codes, audio_len, context_emb=None):
        audio_mask = get_mask_from_lengths(audio_len)
        text_mask = get_mask_from_lengths(text_len)
        # [batch_size, text_len, hidden_dim]
        text_emb = self.text_emb(text)
        text_emb = text_emb * rearrange(text_mask, "B T -> B T 1")
        text_emb = rearrange(text_emb, "B T D -> B D T")

        if self.downsample_layer is not None:
            text_emb, text_len, _ = self.downsample_layer(text=text, text_emb=text_emb, text_len=text_len)
            text_mask = get_mask_from_lengths(text_len)

        attn_mask = rearrange(audio_mask, "B T -> B 1 T 1") * rearrange(text_mask, "B T -> B 1 1 T")
        # Aligner requires an inverted mask
        aligner_text_mask = ~rearrange(text_mask, "B T -> B T 1")

        if context_emb is not None:
            context_emb = rearrange(context_emb, 'B D -> B 1 D')

        text_max_len = text_emb.shape[2]
        audio_max_len = audio_codes.shape[2]
        attn_prior = self._create_alignment_prior(
            text_len=text_len,
            text_max_len=text_max_len,
            audio_len=audio_len,
            audio_max_len=audio_max_len,
        )
        # [batch_size, 1, audio_len, text_len]
        attn_soft, attn_logprob = self.alignment_encoder(
            queries=audio_codes, keys=text_emb, mask=aligner_text_mask, attn_prior=attn_prior, conditioning=context_emb
        )
        attn_soft = attn_soft * attn_mask
        attn_logprob = attn_logprob * attn_mask
        attn_hard = binarize_attention_parallel(attn=attn_soft, in_lens=text_len, out_lens=audio_len)

        durations = attn_hard.sum(2)
        durations = rearrange(durations, 'B 1 T -> B T')

        return durations, text_len, attn_hard, attn_soft, attn_logprob


class TextDownSampling(NeuralModule):

    def __init__(self, input_dim, down_sample_rate, bos_id, eos_id, space_id, space_dur):
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.space_id = space_id
        self.space_dur = space_dur
        self.down_sample_rate = down_sample_rate
        kernel_size = 2 * self.down_sample_rate - 1
        self.downsample_layer = Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            stride=self.down_sample_rate,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), EncodedRepresentation()),
            "text_emb": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "text_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "outputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "output_len": NeuralType(tuple('B'), LengthsType()),
            "text_durs": NeuralType(('B', 'T'), LengthsType()),
        },
    )
    def forward(self, text, text_emb, text_len):
        text_mask = get_mask_from_lengths(text_len)
        is_bos_eos = torch.logical_or(text == self.bos_id, text == self.eos_id)
        is_space = text == self.space_id
        text_durs = torch.where(is_bos_eos, self.down_sample_rate * torch.ones_like(text), torch.ones_like(text))
        text_durs = torch.where(is_space, self.space_dur * torch.ones_like(text), text_durs)
        text_durs = text_durs * text_mask
        text_emb = rearrange(text_emb, 'B D T -> B T D')
        text_emb_repeated, output_len = regulate_len(durations=text_durs, enc_out=text_emb)
        text_emb_repeated = rearrange(text_emb_repeated, 'B T D -> B D T')
        output_len = torch.ceil(output_len / self.down_sample_rate).int()
        out_mask = get_mask_from_lengths(output_len)
        outputs = self.downsample_layer(inputs=text_emb_repeated, mask=out_mask)
        return outputs, output_len, text_durs


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
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, audio_codes, audio_len):
        batch_size = audio_codes.size(0)
        mask = get_mask_from_lengths(audio_len)

        context = self.pre_conv1(inputs=audio_codes, mask=mask)
        context = self.pre_conv2(inputs=context, mask=mask)

        context = rearrange(context, 'B D T -> B T D')
        context_input_emb = self.input_emb.tile([batch_size, 1, 1])

        context_input_len = audio_len + 1
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
        mask_min=0.0,
        mask_max=0.0,
        mask_alpha=1.0,
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

        if mask_max > 0.0:
            self.text_masking = FeatureMasking(
                mask_min=mask_min,
                mask_max=mask_max,
                mask_alpha=mask_alpha
            )
        else:
            self.text_masking = None

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), EncodedRepresentation()),
            "text_len": NeuralType(('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "out_len": NeuralType(tuple('B'), LengthsType()),
            "text_durs": NeuralType(('B', 'T'), LengthsType()),
        }

    @typecheck()
    def forward(self, text, text_len, context_emb):
        text_mask = get_mask_from_lengths(text_len)
        text_emb = self.word_emb(text)

        if self.training and self.text_masking is not None:
            text_emb = self.text_masking.apply_dropout(inputs=text_emb, input_len=text_len)

        out = self.transformer(x=text_emb, x_mask=text_mask)['output']

        if self.downsample_layer is not None:
            out = rearrange(out, 'B T D -> B D T')
            out, out_len, text_durs = self.downsample_layer(text=text, text_emb=out, text_len=text_len)
            out = rearrange(out, 'B D T -> B T D')
        else:
            out_len = text_len
            text_durs = torch.ones_like(text)
            text_durs = text_durs * text_mask

        out_mask = get_mask_from_lengths(out_len)

        context_emb = rearrange(context_emb, 'B D -> B 1 D')
        context_res = self.context_cond_layer(context_emb)
        out = out + context_res
        out = out * rearrange(out_mask, 'B T -> B T 1')

        return out, out_len, text_durs


class DurationDecoder(NeuralModule):

    def __init__(
        self,
        parallel_transformer,
        duration_transformer,
        input_dim,
        d_model,
        num_duration,
        mask_min=0.0,
        mask_max=0.9,
    ):
        super(DurationDecoder, self).__init__()
        self.d_model = d_model
        self.num_duration = num_duration
        self.parallel_transformer = parallel_transformer
        self.duration_transformer = duration_transformer

        self.input_layer = torch.nn.Linear(input_dim, d_model)
        self.speaking_rate_cond_layer = torch.nn.Linear(1, d_model)
        self.duration_cond_layer = torch.nn.Linear(1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.duration_layer = torch.nn.Linear(self.d_model, self.num_duration)

        self.layer_norm_parallel = torch.nn.LayerNorm(self.d_model)
        self.duration_layer_parallel = torch.nn.Linear(self.d_model, self.num_duration)

        self.duration_masking = FeatureMasking(mask_min=mask_min, mask_max=mask_max)

    def _compute_logits(self, inputs, dur_mask, layer_norm, projection, topk=None, temperature=None):
        dur_mask_3d = rearrange(dur_mask, 'B T -> B T 1')
        # [B, T, num_codes]
        out = layer_norm(inputs)
        dur_logits = projection(out)
        dur_logits = dur_logits * dur_mask_3d

        # [B, T]
        if topk is None:
            dur_indices_pred = dur_logits.max(dim=2).indices
        else:
            dur_indices_pred = sample_tokens(logits=dur_logits, topk=topk, temperature=temperature)

        dur_indices_pred = dur_indices_pred * dur_mask
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        return dur_indices_pred, dur_logits

    def _forward_parallel(self, inputs, dur_mask, speaking_rate):
        speaking_rate = rearrange(speaking_rate, 'B -> B 1 1')
        # [B, T, hidden_dim]
        sr_res = self.speaking_rate_cond_layer(speaking_rate)

        hidden_state = self.input_layer(inputs)
        hidden_state = hidden_state + sr_res
        hidden_state = hidden_state * rearrange(dur_mask, 'B T -> B T 1')

        hidden_state = self.parallel_transformer(x=hidden_state, x_mask=dur_mask)['output']

        dur_indices_pred, dur_logits = self._compute_logits(
            inputs=hidden_state,
            dur_mask=dur_mask,
            layer_norm=self.layer_norm_parallel,
            projection=self.duration_layer_parallel,
        )

        return dur_indices_pred, dur_logits, hidden_state

    def _forward_duration(self, inputs, dur_len, dur_indices, cond_mask=None, topk=None, temperature=None):
        dur_mask = get_mask_from_lengths(dur_len)
        dur_mask_3d = rearrange(dur_mask, 'B T -> B T 1')

        dur_indices_shifted = dur_indices[:, :-1]
        dur_indices_shifted = torch.nn.functional.pad(dur_indices_shifted, pad=(1, 0))

        log_dur = torch.log(dur_indices_shifted + 1.0).detach()
        log_dur = rearrange(log_dur, 'B T -> B T 1')
        dur_res = self.duration_cond_layer(log_dur)

        if cond_mask is not None:
            dur_res = self.duration_masking(inputs=dur_res, mask=cond_mask)
        elif self.training:
            dur_res = self.duration_masking.apply_dropout(inputs=dur_res, input_len=dur_len)

        hidden_state = inputs + dur_res
        hidden_state = hidden_state * dur_mask_3d

        # [B, T, D]
        hidden_state = self.duration_transformer(x=hidden_state, x_mask=dur_mask)['output']

        dur_indices_pred, dur_logits = self._compute_logits(
            inputs=hidden_state,
            dur_mask=dur_mask,
            layer_norm=self.layer_norm,
            projection=self.duration_layer,
            topk=topk,
            temperature=temperature,
        )
        return dur_indices_pred, dur_logits

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "dur_len": NeuralType(tuple('B'), LengthsType()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
        }

    @property
    def output_types(self):
        return {
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'C', 'T_text'), LogitsType()),
            "dur_indices_pred_parallel": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits_parallel": NeuralType(('B', 'C', 'T_text'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, dur_len, dur_indices, speaking_rate):
        dur_mask = get_mask_from_lengths(dur_len)

        dur_indices_pred_parallel, dur_logits_parallel, hidden_state = self._forward_parallel(
            inputs=inputs, dur_mask=dur_mask, speaking_rate=speaking_rate,
        )

        dur_indices_pred, dur_logits = self._forward_duration(
            inputs=hidden_state, dur_len=dur_len, dur_indices=dur_indices,
        )

        return dur_indices_pred, dur_logits, dur_indices_pred_parallel, dur_logits_parallel

    def infer(
        self,
        inputs,
        dur_len,
        speaking_rate,
        frames_per_iter,
        infer_weight=1.0,
        topk=None,
        temperature=None,
        silence_pad_start=None,
        silence_pad_end=None,
    ):
        ar_weight = infer_weight / (1.0 + infer_weight)
        parallel_weight = 1.0 / (1.0 + infer_weight)
        batch_size = inputs.shape[0]
        # [B, T]
        dur_mask = get_mask_from_lengths(dur_len, pad_to_factor=frames_per_iter)

        dur_len_padded = torch.ceil(dur_len / frames_per_iter).int() * frames_per_iter
        max_len = dur_len.max()
        max_len_padded = dur_len_padded.max()
        inputs = torch.nn.functional.pad(inputs, (0, 0, 0, max_len_padded - max_len))

        # [B, T]
        dur_indices_shape = [batch_size, max_len_padded]
        dur_indices = torch.zeros(dur_indices_shape, dtype=torch.int, device=inputs.device)

        _, logits_parallel, hidden_state_input = self._forward_parallel(
            inputs=inputs,
            dur_mask=dur_mask,
            speaking_rate=speaking_rate,
        )
        logits_parallel = rearrange(logits_parallel, 'B C T -> B T C')

        self.duration_transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)

        for i in range(0, max_len_padded, frames_per_iter):
            len_i = i + frames_per_iter
            dur_len_i = torch.clamp_max(dur_len_padded, max=len_i)
            hidden_state_i = hidden_state_input[:, :len_i, :]
            dur_indices_i = dur_indices[:, :len_i]
            dur_mask_i = dur_mask[:, : len_i]

            dur_cond_mask_i = torch.zeros_like(dur_mask_i)
            for j in range(1, frames_per_iter):
                dur_cond_mask_i[:, i + j] = True

            # [B, C, T], [B, C, W, T]
            _, logits_i = self._forward_duration(
                inputs=hidden_state_i,
                dur_len=dur_len_i,
                cond_mask=dur_cond_mask_i,
                dur_indices=dur_indices_i,
            )
            logits_i = rearrange(logits_i, 'B C T -> B T C')
            logits = (parallel_weight * logits_parallel[:, :i + frames_per_iter]) + (ar_weight * logits_i)

            if topk is None:
                dur_indices_i = logits.max(dim=2).indices
            else:
                dur_indices_i = sample_tokens(logits=logits, topk=topk, temperature=temperature)

            for j in range(frames_per_iter):
                dur_indices[:, i + j] = dur_indices_i[:, i + j]

            if i == 0 and silence_pad_start:
                dur_indices[:, 0] = silence_pad_start - 1

        dur_indices = dur_indices[:, :max_len_padded]

        if silence_pad_end:
            for i in range(dur_indices.shape[0]):
                last_i = dur_len[i] - 1
                dur_indices[i, last_i] = silence_pad_end - 1

        dur_indices = dur_indices * dur_mask
        dur_indices = dur_indices[:, :max_len]

        self.duration_transformer.reset_cache(use_cache=False)

        return dur_indices


class AudioInputLayer(NeuralModule):

    def __init__(self, input_dim, output_dim, audio_mask_min=0.0, audio_mask_max=0.9):
        super(AudioInputLayer, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, output_dim)
        self.output_layer = torch.nn.Linear(output_dim, output_dim)

        self.masking = FeatureMasking(mask_min=audio_mask_min, mask_max=audio_mask_max)

    def forward(self, hidden_state, audio_codes, audio_len, cond_mask=None):
        audio_mask = get_mask_from_lengths(audio_len)

        res = self.hidden_layer(audio_codes)
        res = self.output_layer(res)

        if cond_mask is not None:
            res = self.masking.forward(inputs=res, mask=cond_mask)
        elif self.training:
            res = self.masking.apply_dropout(inputs=res, input_len=audio_len)

        out = hidden_state + res
        out = out * rearrange(audio_mask, 'B T -> B T 1')

        return out


class AudioPredictionLayer(NeuralModule):

    def __init__(self, input_dim, num_codebooks, codebook_size):
        super(AudioPredictionLayer, self).__init__()
        num_logits = num_codebooks * codebook_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.output_layer =torch.nn.Linear(input_dim, num_logits)

    def forward(self, hidden_state, audio_mask, topk=None, temperature=None):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        # [batch_size, audio_len, num_codebook * codebook_size]
        out = self.layer_norm(hidden_state)
        audio_logits = self.output_layer(out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.num_codebooks, self.codebook_size)
        audio_logits = torch.reshape(audio_logits, logit_shape)

        # [batch_size, audio_len, num_codebook]
        if topk is None:
            audio_tokens = audio_logits.max(dim=3).indices
        else:
            audio_tokens = sample_tokens(logits=audio_logits, topk=topk, temperature=temperature)

        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits


class AcousticLayer(NeuralModule):

    def __init__(
        self,
        input_dim,
        d_model,
        num_codebook,
        codebook_size,
        transformer_kwargs,
    ):
        super(AcousticLayer, self).__init__()
        self.input_layer = AudioInputLayer(input_dim=input_dim, output_dim=d_model)
        self.transformer = Transformer(**transformer_kwargs)
        self.predict_layer = AudioPredictionLayer(
            input_dim=d_model, num_codebooks=num_codebook, codebook_size=codebook_size
        )

    def forward(self, hidden_state, audio_len, audio_codes, condition_input=False, cond_mask=None):
        audio_mask = get_mask_from_lengths(audio_len)
        if condition_input:
            hidden_state = self.input_layer(
                hidden_state=hidden_state, audio_codes=audio_codes, audio_len=audio_len, cond_mask=cond_mask
            )
        hidden_state = self.transformer(x=hidden_state, x_mask=audio_mask)['output']
        audio_tokens, audio_logits = self.predict_layer(hidden_state=hidden_state, audio_mask=audio_mask)
        return hidden_state, audio_tokens, audio_logits



class AudioDecoder(NeuralModule):

    def __init__(
        self,
        parallel_transformer,
        semantic_transformer,
        acoustic_transformer_kwargs,
        input_dim,
        d_model,
        num_acoustic_codebooks,
        codebook_size,
        codebook_dim,
    ):
        super(AudioDecoder, self).__init__()
        self.num_codebooks = num_acoustic_codebooks + 1
        self.codebook_dim = codebook_dim
        self.codebook_emb_dim = codebook_dim * self.num_codebooks

        self.input_layer = torch.nn.Linear(input_dim, d_model)
        self.parallel_transformer = parallel_transformer
        self.semantic_parallel_predict_layer = AudioPredictionLayer(
            input_dim=d_model, num_codebooks=1, codebook_size=codebook_size
        )

        self.audio_input_layer = AudioInputLayer(input_dim=self.codebook_emb_dim, output_dim=d_model)
        self.semantic_transformer = semantic_transformer
        self.semantic_predict_layer = AudioPredictionLayer(
            input_dim=d_model, num_codebooks=1, codebook_size=codebook_size
        )

        self.acoustic_layers = torch.nn.ModuleList()
        for _ in range(num_acoustic_codebooks):
            acoustic_layer = AcousticLayer(
                input_dim=codebook_dim,
                d_model=d_model,
                num_codebook=1,
                codebook_size=codebook_size,
                transformer_kwargs=acoustic_transformer_kwargs,
            )
            self.acoustic_layers.append(acoustic_layer)

    def _forward_parallel(self, hidden_state, audio_len):
        audio_mask = get_mask_from_lengths(audio_len)
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        hidden_state = self.input_layer(hidden_state)
        hidden_state = hidden_state * audio_mask_3d
        hidden_state = self.parallel_transformer(x=hidden_state, x_mask=audio_mask)['output']
        semantic_tokens_parallel, semantic_logits_parallel = self.semantic_parallel_predict_layer(
            hidden_state=hidden_state, audio_mask=audio_mask, topk=None, temperature=None
        )

        return semantic_tokens_parallel, semantic_logits_parallel, hidden_state

    def _forward_semantic(self, hidden_state, audio_len, audio_codes, topk=None, temperature=None, cond_mask=None):
        audio_mask = get_mask_from_lengths(audio_len)

        audio_codes_shifted = audio_codes[:, :-1, :]
        audio_codes_shifted = torch.nn.functional.pad(audio_codes_shifted, pad=(0, 0, 1, 0))

        # [batch_size, audio_len, hidden_dim]
        hidden_state = self.audio_input_layer(
            hidden_state=hidden_state, audio_codes=audio_codes_shifted, audio_len=audio_len, cond_mask=cond_mask,
        )
        hidden_state = self.semantic_transformer(x=hidden_state, x_mask=audio_mask)['output']
        semantic_tokens, semantic_logits = self.semantic_predict_layer(
            hidden_state=hidden_state, audio_mask=audio_mask, topk=topk, temperature=temperature
        )

        return semantic_tokens, semantic_logits, hidden_state

    def _forward_acoustic(self, hidden_state, audio_len, audio_codes):
        audio_token_list = []
        audio_logit_list = []
        for i, acoustic_layer in enumerate(self.acoustic_layers):
            start_i = i * self.codebook_dim
            end_i = (i + 1) * self.codebook_dim
            audio_codes_i = audio_codes[:, :, start_i:end_i]
            hidden_state, audio_tokens, audio_logits = acoustic_layer(
                hidden_state=hidden_state, audio_codes=audio_codes_i, audio_len=audio_len, condition_input=True
            )
            audio_token_list.append(audio_tokens)
            audio_logit_list.append(audio_logits)

        audio_tokens = torch.cat(audio_token_list, dim=1)
        audio_logits = torch.cat(audio_logit_list, dim=1)

        return audio_tokens, audio_logits

    def _infer_acoustic(
        self, hidden_state, audio_len, semantic_codes, vector_quantizer, cond_layers, cond_mask=None
    ):
        audio_token_list = []
        input_codes = semantic_codes
        for i, acoustic_layer in enumerate(self.acoustic_layers):
            cond_input = i in cond_layers
            hidden_state, audio_tokens_i, _ = acoustic_layer(
                hidden_state=hidden_state,
                audio_codes=input_codes,
                audio_len=audio_len,
                condition_input=cond_input,
                cond_mask=cond_mask
            )
            audio_token_list.append(audio_tokens_i)

            audio_tokens_rearrange_i = rearrange(audio_tokens_i, 'B C T -> C B T')
            # [B, D, T]
            input_codes = vector_quantizer.decode(indices=audio_tokens_rearrange_i, input_len=audio_len)
            input_codes = rearrange(input_codes, 'B D T -> B T D')

        audio_tokens = torch.cat(audio_token_list, dim=1)

        return audio_tokens

    @property
    def input_types(self):
        return {
            "hidden_state": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
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
    def forward(self, hidden_state, audio_len, audio_codes):
        semantic_tokens_parallel, semantic_logits_parallel, hidden_state = self._forward_parallel(
            hidden_state=hidden_state, audio_len=audio_len,
        )
        semantic_tokens, semantic_logits, hidden_state = self._forward_semantic(
            hidden_state=hidden_state, audio_len=audio_len, audio_codes=audio_codes,
        )
        acoustic_tokens, acoustic_logits = self._forward_acoustic(
            hidden_state=hidden_state, audio_len=audio_len, audio_codes=audio_codes,
        )

        return (semantic_tokens_parallel, semantic_logits_parallel, semantic_tokens, semantic_logits,
                acoustic_tokens, acoustic_logits)

    def infer(
        self,
        inputs,
        audio_len,
        frames_per_iter,
        vector_quantizer,
        infer_weight=1.0,
        topk=None,
        temperature=None,
        cond_layers=None
    ):
        if cond_layers is None:
            cond_layers = set(range(len(self.acoustic_layers)))

        ar_weight = infer_weight / (1.0 + infer_weight)
        parallel_weight = 1.0 / (1.0 + infer_weight)
        batch_size = inputs.shape[0]
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_len, pad_to_factor=frames_per_iter)

        audio_len_padded = torch.ceil(audio_len / frames_per_iter).int() * frames_per_iter
        max_len = audio_len.max()
        max_len_padded = audio_len_padded.max()
        inputs = torch.nn.functional.pad(inputs, (0, 0, 0, max_len_padded - max_len))

        # [B, T, C]
        audio_token_shape = [batch_size, max_len_padded, self.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [batch_size, max_len_padded, self.codebook_emb_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)
        # [B, T, D]
        semantic_code_shape = [batch_size, max_len_padded, self.codebook_dim]
        semantic_codes = torch.zeros(semantic_code_shape, dtype=torch.float, device=inputs.device)

        _, logits_parallel, hidden_state_input = self._forward_parallel(hidden_state=inputs, audio_len=audio_len_padded)
        logits_parallel = rearrange(logits_parallel, 'B C W T -> B T C W')

        self.semantic_transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)
        for acoustic_layer in self.acoustic_layers:
            acoustic_layer.transformer.reset_cache(use_cache=True, frames_per_iter=frames_per_iter)

        for i in range(0, max_len_padded, frames_per_iter):
            len_i = i + frames_per_iter
            audio_len_i = torch.clamp_max(audio_len_padded, max=len_i)
            audio_codes_i = audio_codes[:, : len_i, :]
            audio_mask_i = audio_mask[:, : len_i]
            hidden_state_i = hidden_state_input[:, : len_i, :]

            audio_cond_mask_i = torch.zeros_like(audio_mask_i)
            for j in range(1, frames_per_iter):
                audio_cond_mask_i[:, i + j] = True

            _, logits_i, hidden_state_i = self._forward_semantic(
                hidden_state=hidden_state_i,
                audio_len=audio_len_i,
                audio_codes=audio_codes_i,
                cond_mask=audio_cond_mask_i,
            )
            logits_i = rearrange(logits_i, 'B C W T -> B T C W')
            logits = (parallel_weight * logits_parallel[:, :len_i]) + (ar_weight * logits_i)

            if topk is None:
                semantic_tokens_i = logits.max(dim=3).indices
            else:
                semantic_tokens_i = sample_tokens(logits=logits, topk=topk, temperature=temperature)

            semantic_tokens_rearrange_i = rearrange(semantic_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            semantic_codes_pred_i = vector_quantizer.decode(indices=semantic_tokens_rearrange_i, input_len=audio_len)
            semantic_codes_pred_i = rearrange(semantic_codes_pred_i, 'B D T -> B T D')

            for j in range(frames_per_iter):
                semantic_codes[:, i + j, :] = semantic_codes_pred_i[:, i + j, :]

            semantic_codes_i = semantic_codes[:, : len_i, :]

            acoustic_tokens_i = self._infer_acoustic(
                hidden_state=hidden_state_i,
                audio_len=audio_len_i,
                semantic_codes=semantic_codes_i,
                vector_quantizer=vector_quantizer,
                cond_layers=cond_layers,
                cond_mask=audio_cond_mask_i,
            )
            acoustic_tokens_i = rearrange(acoustic_tokens_i, 'B C T -> B T C')
            acoustic_tokens_rearrange_i = rearrange(acoustic_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            acoustic_codes_pred_i = vector_quantizer.decode(indices=acoustic_tokens_rearrange_i, input_len=audio_len)
            acoustic_codes_pred_i = rearrange(acoustic_codes_pred_i, 'B D T -> B T D')

            for j in range(frames_per_iter):
                audio_codes_pred_i = torch.concat([semantic_codes_pred_i, acoustic_codes_pred_i], dim=2)
                audio_tokens_i = torch.concat([semantic_tokens_i, acoustic_tokens_i], dim=2)
                audio_codes[:, i + j, :] = audio_codes_pred_i[:, i + j, :]
                audio_tokens[:, i + j, :] = audio_tokens_i[:, i + j, :]

        audio_tokens = audio_tokens[:, :max_len, :]
        audio_mask_unpadded = get_mask_from_lengths(audio_len)
        audio_tokens = audio_tokens * audio_mask_unpadded.unsqueeze(2)
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        self.semantic_transformer.reset_cache(use_cache=False)
        for acoustic_layer in self.acoustic_layers:
            acoustic_layer.transformer.reset_cache(use_cache=False)

        return audio_tokens
