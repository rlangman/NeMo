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
from nemo.collections.tts.modules.transformer import PositionalEmbedding
from nemo.collections.tts.modules.acoustic_decoder_transformer import TransformerLayer, TransformerCrossAttentionLayer
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import EncodedRepresentation, LengthsType, MaskType
from nemo.core.neural_types.neural_type import NeuralType


class TextEncoder(NeuralModule):
    def __init__(
        self,
        num_text_emb,
        padding_idx,
        d_context,
        transformer,
    ):
        super(TextEncoder, self).__init__()
        self.d_model = transformer.d_model

        self.text_embeddings = nn.Embedding(num_text_emb, self.d_model, padding_idx=padding_idx)
        self.filler_embedding = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))
        self.positional_embedding = PositionalEmbedding(self.d_model)
        self.context_cond_layer = torch.nn.Linear(d_context, self.d_model)
        self.transformer = transformer

    def _create_padded_input(
        self,
        text,
        text_mask,
        audio_mask,
        context_emb,
    ):
        max_text_len = text_mask.shape[1]
        max_audio_len = audio_mask.shape[1]
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        # [B, T, D]
        text_emb = self.text_embeddings(text)
        text_mask_padded = torch.nn.functional.pad(text_mask, pad=[0, max_audio_len - max_text_len])
        text_emb = torch.nn.functional.pad(text_emb, pad=[0, 0, 0, max_audio_len - max_text_len])

        filler_tokens = torch.ones_like(text_emb, device=text_emb.device) * self.filler_embedding
        filler_mask = torch.logical_xor(audio_mask, text_mask_padded)
        filler_mask = rearrange(filler_mask, 'B T -> B T 1')
        filler_tokens = filler_tokens * filler_mask

        context_emb = rearrange(context_emb, 'B D -> B 1 D')
        context_res = self.context_cond_layer(context_emb)

        pos_seq = torch.arange(max_audio_len, device=text_emb.device).to(text_emb.dtype)
        pos_emb = self.positional_embedding(pos_seq)

        encoder_input = text_emb + filler_tokens + context_res + pos_emb
        encoder_input = encoder_input * audio_mask_3d

        return encoder_input

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), EncodedRepresentation()),
            "text_lens": NeuralType(('B'), LengthsType()),
            "audio_lens": NeuralType(('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, text, text_lens, audio_lens, context_emb, context, context_mask):
        text_mask = get_mask_from_lengths(text_lens)
        audio_mask = get_mask_from_lengths(audio_lens)

        out = self._create_padded_input(text=text, text_mask=text_mask, audio_mask=audio_mask, context_emb=context_emb)
        out = self.transformer(inputs=out, audio_mask=audio_mask, context=context, context_mask=context_mask)

        return out


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
        self.transformer_layers = nn.ModuleList([
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
        ])

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
            out = layer(
                inputs=out,
                mask=text_mask,
                encoded=context,
                attn_mask=context_attn_mask
            )

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

        dur_enc = self.transformer(
            inputs=dur_enc, text_mask=text_mask, context=context, context_mask=context_mask
        )

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
        self.transformer_layers = nn.ModuleList([
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
        ])

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
            out = layer(
                inputs=out,
                mask=audio_mask,
                encoded=context,
                attn_mask=context_attn_mask
            )

        out = out * audio_mask_3d

        return out


class AudioEncoder(NeuralModule):
    def __init__(self, input_dim, transformer):
        super(AudioEncoder, self).__init__()

        audio_hidden_dim = transformer.d_model
        self.input_layer = torch.nn.Linear(input_dim, audio_hidden_dim)
        self.positional_embedding = PositionalEmbedding(audio_hidden_dim)
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, audio_mask, context, context_mask):
        audio_enc = self.input_layer(inputs)

        max_audio_len = audio_mask.shape[1]
        pos_seq = torch.arange(max_audio_len, device=audio_enc.device).to(audio_enc.dtype)
        pos_emb = self.positional_embedding(pos_seq)

        audio_enc = audio_enc + pos_emb
        audio_enc = audio_enc * rearrange(audio_mask, 'B T -> B T 1')
        audio_enc = self.transformer(
            inputs=audio_enc, audio_mask=audio_mask, context=context, context_mask=context_mask
        )

        return audio_enc