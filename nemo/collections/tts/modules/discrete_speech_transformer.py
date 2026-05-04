# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from nemo.collections.tts.modules.acoustic_model_modules import TextDownSampling
from nemo.collections.tts.modules.acoustic_model_transformer import TransformerCrossAttentionLayer, TransformerLayer
from nemo.collections.tts.modules.transformer import PositionalEmbedding
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import EncodedRepresentation, LengthsType, MaskType
from nemo.core.neural_types.neural_type import NeuralType


class TextEncoder(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        n_embed,
        d_model,
        d_context,
        d_head,
        d_inner,
        kernel_size,
        padding_idx,
        down_sample_rate,
        bos_id,
        eos_id,
        space_id,
        space_dur,
        dropout=0.1,
        dropout_att=0.1,
    ):
        super(TextEncoder, self).__init__()
        self.d_model = d_model

        self.word_emb = nn.Embedding(n_embed, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.context_cond_layer = torch.nn.Linear(d_context, self.d_model)
        self.text_layers = nn.ModuleList(
            [
                TransformerLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    kernel_size,
                    dropout,
                    dropatt=dropout_att,
                )
                for _ in range(n_layer)
            ]
        )
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

        pos_seq = torch.arange(text_emb.size(1), device=text_emb.device).to(text_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)

        out = text_emb + pos_emb
        out = out * rearrange(text_mask, 'B T -> B T 1')

        for layer in self.text_layers:
            out = layer(out, mask=text_mask)

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


class DurationTransformer(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head_self,
        n_head_context,
        d_model,
        d_context,
        d_head,
        d_inner,
        kernel_size=None,
        dropout=0.1,
        dropout_self_att=0.1,
        dropout_context_att=0.1,
    ):
        super(DurationTransformer, self).__init__()
        self.d_model = d_model
        self.transformer_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    n_head_self=n_head_self,
                    n_head_cross=n_head_context,
                    d_model=d_model,
                    d_encoded=d_context,
                    d_head=d_head,
                    d_inner=d_inner,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dropout_self_att=dropout_self_att,
                    dropout_cross_att=dropout_context_att,
                )
                for _ in range(n_layer)
            ]
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_input'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, text_mask, context, context_mask):
        text_mask_3d = rearrange(text_mask, 'B T_text -> B T_text 1')

        max_context_len = context.shape[1]
        # [B, T_text, T_context]
        context_attn_mask = text_mask_3d.repeat([1, 1, max_context_len])
        context_attn_mask = context_attn_mask * rearrange(context_mask, 'B T_context -> B 1 T_context')

        out = inputs
        for layer in self.transformer_layers:
            out = layer(inputs=out, mask=text_mask, encoded=context, attn_mask=context_attn_mask)

        out = out * text_mask_3d

        return out


class DurationTransformerWithoutCond(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size=None,
        dropout=0.1,
        dropout_att=0.1,
    ):
        super(DurationTransformerWithoutCond, self).__init__()
        self.d_model = d_model
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    n_head=n_head,
                    d_model=d_model,
                    d_head=d_head,
                    d_inner=d_inner,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dropatt=dropout_att,
                )
                for _ in range(n_layer)
            ]
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_input'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, text_mask):
        text_mask_3d = rearrange(text_mask, 'B T_text -> B T_text 1')

        out = inputs
        for layer in self.transformer_layers:
            out = layer(inputs=out, mask=text_mask)

        out = out * text_mask_3d

        return out


class DurationEncoder(NeuralModule):
    def __init__(self, input_dim, transformer):
        super(DurationEncoder, self).__init__()

        hidden_dim = transformer.d_model
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.speaking_rate_cond_layer = torch.nn.Linear(1, hidden_dim)
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

        dur_enc = self.transformer(inputs=dur_enc, text_mask=text_mask, context=context, context_mask=context_mask)

        return dur_enc


class AudioTransformer(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head_self,
        n_head_context,
        d_model,
        d_context,
        d_head,
        d_inner,
        kernel_size=None,
        dropout=0.1,
        dropout_self_att=0.1,
        dropout_context_att=0.1,
    ):
        super(AudioTransformer, self).__init__()
        self.d_model = d_model
        self.transformer_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    n_head_self=n_head_self,
                    n_head_cross=n_head_context,
                    d_model=d_model,
                    d_encoded=d_context,
                    d_head=d_head,
                    d_inner=d_inner,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dropout_self_att=dropout_self_att,
                    dropout_cross_att=dropout_context_att,
                )
                for _ in range(n_layer)
            ]
        )

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_input'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, audio_mask, context, context_mask):
        audio_mask_3d = rearrange(audio_mask, 'B T_audio -> B T_audio 1')

        max_context_len = context.shape[1]
        # [B, T_text, T_context]
        context_attn_mask = audio_mask_3d.repeat([1, 1, max_context_len])
        context_attn_mask = context_attn_mask * rearrange(context_mask, 'B T_context -> B 1 T_context')

        out = inputs
        for layer in self.transformer_layers:
            out = layer(inputs=out, mask=audio_mask, encoded=context, attn_mask=context_attn_mask)

        out = out * audio_mask_3d

        return out


class AudioTransformerWithoutCond(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size=None,
        dropout=0.1,
        dropout_att=0.1,
    ):
        super(AudioTransformerWithoutCond, self).__init__()
        self.d_model = d_model
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    n_head=n_head,
                    d_model=d_model,
                    d_head=d_head,
                    d_inner=d_inner,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dropatt=dropout_att,
                )
                for _ in range(n_layer)
            ]
        )

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_input'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, audio_mask):
        audio_mask_3d = rearrange(audio_mask, 'B T_audio -> B T_audio 1')

        out = inputs
        for layer in self.transformer_layers:
            out = layer(inputs=out, mask=audio_mask)

        out = out * audio_mask_3d

        return out


class AudioEncoder(NeuralModule):
    def __init__(self, input_dim, transformer):
        super(AudioEncoder, self).__init__()

        audio_hidden_dim = transformer.d_model
        self.input_layer = torch.nn.Linear(input_dim, audio_hidden_dim)
        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, audio_hidden_dim]))
        self.positional_embedding = PositionalEmbedding(audio_hidden_dim)
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

        max_audio_len = audio_mask.shape[1]
        pos_seq = torch.arange(max_audio_len, device=audio_enc.device).to(audio_enc.dtype)
        pos_emb = self.positional_embedding(pos_seq)

        audio_enc = audio_enc + pos_emb
        audio_enc = audio_enc * rearrange(audio_mask, 'B T -> B T 1')
        audio_enc = self.transformer(
            inputs=audio_enc, audio_mask=audio_mask, context=context, context_mask=context_mask
        )

        return audio_enc
