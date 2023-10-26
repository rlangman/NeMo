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
import torch.nn.functional as F

from einops import rearrange
from nemo.collections.tts.modules.discrete_speech_modules import DropoutWithoutScaling
from nemo.collections.tts.modules.transformer import PositionalEmbedding
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, regulate_len
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import EncodedRepresentation, LengthsType, MaskType
from nemo.core.neural_types.neural_type import NeuralType


class LinearFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(LinearFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.input_layer = nn.Linear(d_model, d_inner)
        self.activation = nn.ELU()
        self.out_layer = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        res = self.input_layer(inputs)
        res = self.activation(res)
        res = self.out_layer(res)
        res = self.dropout(res)

        output = self.layer_norm(inputs + res)
        output = output.to(inputs.dtype)

        return output


class ConvNeXtFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout):
        super(ConvNeXtFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        padding = (kernel_size // 2)
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        self.layer_norm = nn.LayerNorm(d_model)
        self.inner_layer = nn.Linear(in_features=d_model, out_features=d_inner)
        self.activation = nn.ELU()
        self.skip_layer = nn.Linear(in_features=d_inner, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        res = rearrange(inputs, 'B T D -> B D T')
        res = self.conv(res)
        res = rearrange(res, 'B D T -> B T D')
        res = self.layer_norm(res)
        res = self.inner_layer(res)
        res = self.activation(res)
        res = self.skip_layer(res)
        res = self.dropout(res)
        out = inputs + res

        return out


class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0.1,
    ):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
        residual = inputs

        inputs = self.layer_norm(inputs)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inputs), 3, dim=2)

        head_q = head_q.view(inputs.size(0), inputs.size(1), n_head, d_head)
        head_k = head_k.view(inputs.size(0), inputs.size(1), n_head, d_head)
        head_v = head_v.view(inputs.size(0), inputs.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inputs.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inputs.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inputs.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if mask is not None:
            # [B, 1, T]
            attn_mask = rearrange(~mask, 'B T -> B 1 T')
            # [B * n_head, T, T]
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_mask = attn_mask.to(attn_score.dtype)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inputs.size(0), inputs.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inputs.size(0), inputs.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        output = residual + attn_out

        output = output * rearrange(mask, 'B T -> B T 1')

        return output


class CrossAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_encoded,
        d_head,
        dropout,
        dropatt,
    ):
        super(CrossAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)

        self.query_layer = nn.Linear(d_model, n_head * d_head)
        self.key_layer = nn.Linear(d_encoded, n_head * d_head)
        self.value_layer = nn.Linear(d_encoded, n_head * d_head)
        self.dropout = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.out_layer = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, inputs, encoded, mask, attn_mask):
        # Inputs: [B, T1, D]
        # Encoded: [B, T2, D]
        # Mask: [B, T1]
        # Attn Mask: [B, T1, T2]
        batch_size = inputs.size(0)
        input_max_len = inputs.size(1)
        encoded_max_len = encoded.size(1)

        n_head, d_head = self.n_head, self.d_head

        inputs = self.layer_norm(inputs)

        # [B, T, num_head * head_dim]
        head_q = self.query_layer(inputs)
        head_k = self.key_layer(encoded)
        head_v = self.value_layer(encoded)

        # [B, T, num_head, head_dim]
        head_q = head_q.view(batch_size, input_max_len, n_head, d_head)
        head_k = head_k.view(batch_size, encoded_max_len, n_head, d_head)
        head_v = head_v.view(batch_size, encoded_max_len, n_head, d_head)

        head_q = rearrange(head_q, 'B T H D -> H B T D')
        head_k = rearrange(head_k, 'B T H D -> H B T D')
        head_v = rearrange(head_v, 'B T H D -> H B T D')

        # [num_head * B, T1, head_dim]
        q = head_q.reshape(-1, input_max_len, d_head)
        # [num_head * B, T2, head_dim]
        k = head_k.reshape(-1, encoded_max_len, d_head)
        v = head_v.reshape(-1, encoded_max_len, d_head)

        # [num_head * B, head_Dim, T2]
        k = rearrange(k, 'H T D -> H D T')

        # [num_head * B, T1, T2]
        attn_score = torch.bmm(q, k)
        attn_score.mul_(self.scale)

        # [n_head * B, T1, T2]
        attn_mask = attn_mask.repeat(n_head, 1, 1)
        attn_score.masked_fill_(~attn_mask, -float('inf'))

        # [num_head * B, T1, T2]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = torch.nan_to_num(attn_prob, nan=0.0)
        attn_prob = self.dropatt(attn_prob)
        # [num_head * B, T1, head_dim]
        attn_vec = torch.bmm(attn_prob, v)

        # [num_head, B, T1, head_dim]
        attn_vec = attn_vec.view(n_head, batch_size, input_max_len, d_head)
        attn_vec = rearrange(attn_vec, 'H B T D -> B T H D')
        # [B, T1, num_head * head_dim]
        attn_vec = attn_vec.contiguous().view(batch_size, input_max_len, n_head * d_head)

        # [B, T1, D]
        res = self.out_layer(attn_vec)
        res = self.dropout(res)

        output = inputs + res

        output = output * rearrange(mask, 'B T -> B T 1')

        return output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
    ):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=dropatt,
        )
        if kernel_size == 1:
            self.pos_ff = LinearFF(d_model, d_inner, dropout)
        else:
            self.pos_ff = ConvNeXtFF(d_model, d_inner, kernel_size, dropout)

    def forward(self, dec_inp, mask):
        output = self.dec_attn(inputs=dec_inp, mask=mask)
        output = self.pos_ff(inputs=output)
        output = output * rearrange(mask, 'B T -> B T 1')
        return output


class TransformerCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        n_head_self,
        n_head_cross,
        d_model,
        d_encoded,
        d_head,
        d_inner,
        dropout,
        dropout_self_att,
        dropout_cross_att,
        kernel_size,
    ):
        super(TransformerCrossAttentionLayer, self).__init__()

        self.self_attn = MultiHeadAttn(
            n_head=n_head_self,
            d_model=d_model,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropout_self_att,
        )
        self.cross_attn = CrossAttn(
            n_head=n_head_cross,
            d_model=d_model,
            d_encoded=d_encoded,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropout_cross_att,
        )

        if kernel_size == 1:
            self.pos_ff = LinearFF(d_model, d_inner, dropout)
        else:
            self.pos_ff = ConvNeXtFF(d_model, d_inner, kernel_size, dropout)

    def forward(self, inputs, encoded, mask, attn_mask):
        mask_3d = rearrange(mask, 'B T -> B T 1')
        output = self.self_attn(inputs=inputs, mask=mask)
        output = self.cross_attn(inputs=output, encoded=encoded, mask=mask, attn_mask=attn_mask)
        output = self.pos_ff(output)
        output = output * mask_3d
        return output


class ContextTransformer(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout=0.1,
        dropatt=0.1,
    ):
        super(ContextTransformer, self).__init__()
        self.d_model = d_model

        self.pos_emb = PositionalEmbedding(self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                n_head,
                d_model,
                d_head,
                d_inner,
                kernel_size,
                dropout,
                dropatt=dropatt,
            )
            for _ in range(n_layer)
        ])


    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, inputs, mask):
        mask_3d = rearrange(mask, 'B T -> B T 1')

        pos_seq = torch.arange(inputs.size(1), device=inputs.device).to(inputs.dtype)
        pos_emb = self.pos_emb(pos_seq)

        out = inputs + pos_emb
        out = out * mask_3d

        for layer in self.transformer_layers:
            out = layer(out, mask=mask)

        out = self.layer_norm(out)
        out = out * mask_3d

        return out


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
        dropout=0.1,
        dropout_att=0.1,
        down_sample_rate=2,
        down_sample_kernel_size=3,
    ):
        super(TextEncoder, self).__init__()
        self.d_model = d_model

        self.word_emb = nn.Embedding(n_embed, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.context_cond_layer = torch.nn.Linear(d_context, self.d_model)
        self.text_layers = nn.ModuleList([
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
        ])
        self.down_sample_rate = down_sample_rate
        padding = (down_sample_kernel_size - down_sample_rate + 1) // 2
        self.downsample_layer = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=down_sample_kernel_size, stride=self.down_sample_rate, padding=padding,
        )

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

        out_lens = torch.ceil(text_lens / self.down_sample_rate).int()
        out_mask = get_mask_from_lengths(out_lens)

        out = rearrange(out, 'B T D -> B D T')
        out = self.downsample_layer(out)
        out = rearrange(out, 'B D T -> B T D')

        context_emb = rearrange(context_emb, 'B D -> B 1 D')
        context_res = self.context_cond_layer(context_emb)
        out = out + context_res
        out = out * rearrange(out_mask, 'B T -> B T 1')

        return out, out_lens


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
            "input": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
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
            "text_enc": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "durs": NeuralType(('B', 'T_input'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),

        }

    def forward(self, text_enc, durs, context, context_mask):
        audio_enc = self.input_layer(text_enc)
        audio_enc, audio_lens = regulate_len(durs, audio_enc, pace=1.0)
        audio_mask = get_mask_from_lengths(audio_lens)

        max_audio_len = audio_mask.shape[1]
        pos_seq = torch.arange(max_audio_len, device=audio_enc.device).to(audio_enc.dtype)
        pos_emb = self.positional_embedding(pos_seq)

        audio_enc = audio_enc + pos_emb
        audio_enc = audio_enc * rearrange(audio_mask, 'B T -> B T 1')
        audio_enc = self.transformer(
            inputs=audio_enc, audio_mask=audio_mask, context=context, context_mask=context_mask
        )

        return audio_enc, audio_lens


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
    def __init__(self, input_dim, transformer, speaking_rate_dropout_rate=0.0):
        super(DurationEncoder, self).__init__()

        hidden_dim = transformer.d_model
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.speaking_rate_cond_layer = torch.nn.Linear(1, hidden_dim)
        self.transformer = transformer
        if speaking_rate_dropout_rate:
            self.speaking_rate_dropout = DropoutWithoutScaling(speaking_rate_dropout_rate)
        else:
            self.speaking_rate_dropout = torch.nn.Identity()


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
        sr_res = self.speaking_rate_dropout(sr_res)

        dur_enc = self.input_layer(text_enc)
        dur_enc = dur_enc + sr_res
        dur_enc = dur_enc * rearrange(text_mask, 'B T -> B T 1')

        dur_enc = self.transformer(
            inputs=dur_enc, text_mask=text_mask, context=context, context_mask=context_mask
        )

        return dur_enc
