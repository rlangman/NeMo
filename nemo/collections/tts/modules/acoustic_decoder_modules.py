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

import torch
from einops import rearrange

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
    VoidType
)
from nemo.core.neural_types.neural_type import NeuralType


def get_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def sample_tokens(logits, temperature, topk):
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
        padding = get_padding(kernel_size=kernel_size, stride=stride)
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if activation is None:
            self.activation = None
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "elu":
            self.activation = torch.nn.ELU()
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


class SemanticInputLayer(NeuralModule):

    def __init__(self, input_dim, output_dim):
        super(SemanticInputLayer, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, output_dim)
        self.output_layer = torch.nn.Linear(output_dim, output_dim)

    @property
    def input_types(self):
        return {
            "semantic_codes": NeuralType(('B', 'T', 'C'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'C'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, semantic_codes, audio_mask):
        out = self.hidden_layer(semantic_codes)
        out = self.output_layer(out)
        out = out * rearrange(audio_mask, 'B T -> B T 1')
        return out


class ContextEncoder(NeuralModule):

    def __init__(self, input_dim, encoder):
        super(ContextEncoder, self).__init__()
        d_model = encoder.d_model
        self.pre_conv = Conv1d(
            in_channels=input_dim,
            out_channels=d_model
        )
        self.encoder = encoder

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)
        context = self.pre_conv(inputs=audio_codes, mask=mask)

        context = rearrange(context, 'B D T -> B T D')
        context = self.encoder(inputs=context, mask=mask)
        context = rearrange(context, 'B T D -> B D T')

        return context, audio_lens


class AudioDecoder(NeuralModule):

    def __init__(self, fft, num_codebooks, codebook_size, codebook_dim):
        super(AudioDecoder, self).__init__()
        self.d_model = fft.d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_logits = self.num_codebooks * self.codebook_size

        self.audio_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))
        self.fft = fft

        self.audio_hidden_layer = torch.nn.Linear(codebook_dim, self.d_model )
        self.audio_cond_layer = torch.nn.Linear(self.d_model , self.d_model )

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.audio_token_layer = torch.nn.Linear(self.d_model, self.num_logits)
        self.layer_norm_parallel = torch.nn.LayerNorm(self.d_model)
        self.audio_token_layer_parallel = torch.nn.Linear(self.d_model, self.num_logits)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
            "audio_maskin": NeuralType(('B', 'T_audio'), MaskType()),
            "temperature": NeuralType((), FloatType(), optional=True),
            "topk": NeuralType((), IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
        }

    @typecheck()
    def forward(
        self, inputs, audio_mask, context, context_mask, audio_codes, audio_maskin, temperature=None, topk=None
    ):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        audio_res = self.audio_hidden_layer(audio_codes)
        audio_res = self.audio_cond_layer(audio_res)
        audio_res = audio_res * rearrange(audio_maskin, 'B T -> B T 1')

        masked_mask = ~audio_maskin * audio_mask
        mask_res = self.audio_mask_emb * rearrange(masked_mask, 'B T -> B T 1')

        dec_input = inputs + audio_res + mask_res

        dec_input = dec_input * audio_mask_3d
        dec_out = self.fft(inputs=dec_input, audio_mask=audio_mask, context=context, context_mask=context_mask)

        # [batch_size, audio_len, num_codebook * codebook_size]
        dec_out = self.layer_norm(dec_out)
        audio_logits = self.audio_token_layer(dec_out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.num_codebooks, self.codebook_size)

        audio_logits = torch.reshape(audio_logits, logit_shape)
        # [batch_size, audio_len, num_codebook]
        if temperature is None:
            audio_tokens = audio_logits.max(dim=3).indices
        else:
            audio_tokens = sample_tokens(logits=audio_logits, temperature=temperature, topk=topk)

        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits

    def forward_parallel(self, inputs, audio_mask, temperature=None, topk=None):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        # [batch_size, audio_len, num_codebook * codebook_size]
        out = self.layer_norm_parallel(inputs)
        audio_logits = self.audio_token_layer_parallel(out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.num_codebooks, self.codebook_size)
        audio_logits = torch.reshape(audio_logits, logit_shape)

        # [batch_size, audio_len, num_codebook]
        if temperature is None:
            audio_tokens = audio_logits.max(dim=3).indices
        else:
            audio_tokens = sample_tokens(logits=audio_logits, temperature=temperature, topk=topk)

        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits
