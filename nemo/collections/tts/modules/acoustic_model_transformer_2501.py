# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.tts.modules.acoustic_model_modules import Conv1d, sample_tokens
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


class TextEncoder(NeuralModule):
    def __init__(self, transformer, d_model, n_embed, padding_idx):
        super(TextEncoder, self).__init__()
        self.word_emb = torch.nn.Embedding(n_embed, d_model, padding_idx=padding_idx)
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_audio'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, text, text_mask):
        text_emb = self.word_emb(text)
        text_enc = self.transformer(x=text_emb, x_mask=text_mask)['output']
        return text_enc


class ContextEncoder(NeuralModule):

    def __init__(self, input_dim, d_model, transformer):
        super(ContextEncoder, self).__init__()
        self.pre_conv = Conv1d(in_channels=input_dim, out_channels=d_model)
        self.transformer = transformer

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
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)
        context = self.pre_conv(inputs=audio_codes, mask=mask)

        context = rearrange(context, 'B D T -> B T D')
        context = self.transformer(x=context, x_mask=mask)['output']
        context = rearrange(context, 'B T D -> B D T')

        return context


class AudioEncoderWithText(NeuralModule):
    def __init__(self, transformer):
        super(AudioEncoderWithText, self).__init__()
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "text_enc": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, inputs, audio_mask, text_enc, text_mask):
        audio_enc = self.transformer(
            x=inputs,
            x_mask=audio_mask,
            cond=text_enc,
            cond_mask=text_mask,
        )['output']
        return audio_enc


class AudioEncoderWithContext(NeuralModule):
    def __init__(self, transformer):
        super(AudioEncoderWithContext, self).__init__()
        self.transformer = transformer
        num_layers = len(self.transformer.layers)
        self.multi_encoder_mapping = [0 if i % 2 == 0 else 1 for i in range(num_layers)]

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
            "text_enc": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, inputs, audio_mask, context, context_mask, text_enc, text_mask):
        cond = [text_enc, context]
        cond_mask = [text_mask, context_mask]
        audio_enc = self.transformer(
            x=inputs,
            x_mask=audio_mask,
            cond=cond,
            cond_mask=cond_mask,
            multi_encoder_mapping=self.multi_encoder_mapping,
        )['output']
        return audio_enc


class AudioDecoder(NeuralModule):

    def __init__(self, transformer, d_model, num_codebooks, codebook_size, codebook_dim):
        super(AudioDecoder, self).__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_logits = self.num_codebooks * self.codebook_size

        self.transformer = transformer

        self.audio_hidden_layer = torch.nn.Linear(codebook_dim, d_model)
        self.audio_cond_layer = torch.nn.Linear(d_model, d_model)

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.audio_token_layer = torch.nn.Linear(d_model, self.num_logits)
        self.layer_norm_parallel = torch.nn.LayerNorm(d_model)
        self.audio_token_layer_parallel = torch.nn.Linear(d_model, self.num_logits)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
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
    def forward(self, inputs, audio_mask, audio_codes, temperature=None, topk=None):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        audio_codes_shifted = audio_codes[:, :-1, :]
        audio_codes_shifted = torch.nn.functional.pad(audio_codes_shifted, pad=(0, 0, 1, 0))

        audio_res = self.audio_hidden_layer(audio_codes_shifted)
        audio_res = self.audio_cond_layer(audio_res)
        audio_res = audio_res * audio_mask_3d

        dec_input = inputs + audio_res

        dec_input = dec_input * audio_mask_3d
        dec_out = self.transformer(x=dec_input, x_mask=audio_mask)['output']

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

    def infer(
        self,
        inputs,
        audio_lens,
        vector_quantizer,
        temperature=None,
        topk=None,
    ):
        self.transformer.reset_cache(use_cache=True)

        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens)
        audio_mask_3d = audio_mask.unsqueeze(2)

        # [B, T, C]
        audio_token_shape = [audio_mask.shape[0], audio_mask.shape[1], self.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [audio_mask.shape[0], audio_mask.shape[1] + 1, self.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        max_len = audio_lens.max()
        for i in range(max_len):
            inputs_i = inputs[:, : i + 1, :]
            audio_codes_i = audio_codes[:, : i + 1, :]
            audio_mask_i = audio_mask[:, : i + 1]
            # [B, C, T], [B, C, W, T]
            audio_tokens_i, audio_logits = self.forward(
                inputs=inputs_i,
                audio_mask=audio_mask_i,
                audio_codes=audio_codes_i,
                temperature=temperature,
                topk=topk,
            )
            audio_tokens_i = rearrange(audio_tokens_i, 'B C T -> B T C')
            audio_tokens_rearrange_i = rearrange(audio_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            audio_codes_pred = vector_quantizer.decode(indices=audio_tokens_rearrange_i, input_len=audio_lens)
            audio_codes_pred = rearrange(audio_codes_pred, 'B D T -> B T D')
            audio_codes[:, i + 1, :] = audio_codes_pred[:, i, :]
            audio_tokens[:, i, :] = audio_tokens_i[:, i, :]

        audio_tokens = audio_tokens * audio_mask_3d
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        self.transformer.reset_cache(use_cache=False)

        return audio_tokens
