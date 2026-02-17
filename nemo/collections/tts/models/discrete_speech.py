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

from pathlib import Path
import math
from typing import List

import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.pytorch import Trainer

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.acoustic_decoder_loss import AudioTokenLoss
from nemo.collections.tts.losses.discrete_speech_loss import SpeakingRateLoss
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    BoolType,
    EncodedRepresentation,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    ProbsType,
    TokenDurationType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils
from nemo.utils.decorators import experimental


@experimental
class DiscreteSpeechModel(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = self._create_tokenizer(cfg.text_tokenizer)
        self.pad_with_space = self.text_tokenizer.pad_with_space
        self.inference_phoneme_probability = cfg.get("inference_phoneme_probability", 1.0)

        super().__init__(cfg=cfg, trainer=trainer)

        # Text tokenizer information
        num_text_emb = len(self.text_tokenizer.tokens)
        self.text_pad_token = self.text_tokenizer.pad
        self.space_token = self.text_tokenizer.space

        # Maximum duration of a single biphone
        self.max_token_duration = cfg.get("max_token_duration")

        # Context length in terms of number of audio tokens
        self.context_min_len = cfg.get("context_min_len", 50)
        self.context_max_len = cfg.get("context_max_len", 125)
        self.context_len_noise = cfg.get("context_len_noise", 5)

        # Quantizer definitions
        self.semantic_codebook_num = cfg.get("semantic_codebook_num")
        self.semantic_codebook_dim = cfg.get("semantic_codebook_dim")
        self.vector_quantizer = instantiate(cfg.vector_quantizer)
        self.vector_quantizer_semantic = instantiate(cfg.vector_quantizer_semantic)

        self.speaking_rate_quantizer = instantiate(cfg.speaking_rate_quantizer)

        # Encoder, decoder definitions
        self.text_encoder = instantiate(cfg.text_encoder, num_text_emb=num_text_emb, padding_idx=self.text_pad_token)
        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(cfg.decoder)
        self.speaking_rate_predictor = instantiate(cfg.speaking_rate_predictor)

        if "text_conditionining_layer" in cfg:
            self.text_conditionining_layer = instantiate(cfg.text_conditionining_layer, num_text_emb=num_text_emb)
            # Rate at which noise is added to ground truth alignment seen by the decoder
            self.decoder_duration_noise_percent = cfg.get("decoder_duration_noise_percent", 0.2)
        else:
            self.text_conditionining_layer = None
            self.decoder_duration_noise_percent = None

        # Context encoder definition
        self.context_encoder = instantiate(cfg.context_encoder)

        # Aligner definition
        self.aligner = instantiate(cfg.aligner, num_text_emb=num_text_emb)
        self.decoder_aligner = instantiate(cfg.decoder_aligner, num_text_emb=num_text_emb)

        # Infilling hyperparameters
        self.semantic_infill_min = cfg.get("semantic_infill_min", 0.05)
        self.semantic_infill_max = cfg.get("semantic_infill_max", 0.5)
        semantic_infill_beta = cfg.get("semantic_infill_beta", 2.0)
        self.semantic_infill_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=semantic_infill_beta)

        # Reconstruction losses
        self.audio_token_loss_scale = cfg.get("audio_token_loss_scale", 1.0)
        self.semantic_token_loss_fn = AudioTokenLoss(num_codebooks=self.semantic_codebook_num)

        self.speaking_rate_loss_scale = cfg.get("speaking_rate_loss_scale", 1E-3)
        self.speaking_rate_loss_fn = SpeakingRateLoss()

        # Aligner losses
        self.aligner_bin_loss_scale = cfg.get("aligner_bin_loss_scale", 0.01)
        self.aligner_ctc_loss_scale = cfg.get("aligner_ctc_loss_scale", 0.01)
        self.bin_loss_start_epoch = cfg.get("bin_loss_start_epoch", 0)
        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 10)

        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()

        self.log_config = cfg.get("log_config", None)

    def _create_tokenizer(self, tokenizer_config):
        if "g2p" in tokenizer_config:
            if "phoneme_dict" in tokenizer_config.g2p:
                tokenizer_config.g2p.phoneme_dict = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', tokenizer_config.g2p.phoneme_dict,
                )

            if "heteronyms" in tokenizer_config.g2p:
                tokenizer_config.g2p.heteronyms = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', tokenizer_config.g2p.heteronyms,
                )

        text_tokenizer = instantiate(tokenizer_config)
        return text_tokenizer

    def parse(self, str_input: str) -> torch.tensor:
        if not hasattr(self.text_tokenizer, "set_phone_prob"):
            text_tokens = self.text_tokenizer.encode(str_input)
        else:
            with self.text_tokenizer.set_phone_prob(prob=self.inference_phoneme_probability):
                text_tokens = self.text_tokenizer.encode(str_input)

        token_tensor = torch.tensor(text_tokens).unsqueeze_(0).long().to(self.device)
        return token_tensor

    def create_infill_mask(self, input_lens, dist, infill_min, infill_max):
        batch_size = input_lens.shape[0]
        len_mask = get_mask_from_lengths(input_lens)
        max_len = len_mask.shape[1]

        infill_percent = dist.sample(sample_shape=torch.Size([batch_size])).to(input_lens.device)
        infill_percent = infill_min + (infill_max - infill_min) * infill_percent
        infill_len = (infill_percent * input_lens.float())
        infill_rank = torch.clamp_min(infill_len - 1, 0).long()
        infill_rank = rearrange(infill_rank, 'B -> B 1')

        # [batch_size, time]
        infill_vals = torch.rand(size=len_mask.shape, device=input_lens.device)
        infill_vals = infill_vals * len_mask
        infill_topk = torch.topk(infill_vals, k=max_len, dim=1, sorted=True).values
        infill_min_val = torch.gather(infill_topk, index=infill_rank, dim=1)
        infill_mask = infill_vals >= infill_min_val

        infill_mask = infill_mask * len_mask
        infill_loss_mask = ~infill_mask * len_mask

        return infill_mask, infill_loss_mask

    def _add_decoder_duration_noise(self, durs, text_lens):
        max_text_len = durs.shape[1]
        mask = get_mask_from_lengths(text_lens)

        indices = torch.arange(max_text_len, device=durs.device) + 1
        noise_mask = torch.where(rearrange(indices, 'T -> 1 T') == rearrange(text_lens, 'B -> B 1'),
                                 torch.zeros_like(mask), mask)

        end_indices = torch.cumsum(durs, dim=1)
        end_indices = end_indices * mask

        add_noise = torch.rand(size=durs.shape, device=durs.device) < self.decoder_duration_noise_percent
        noise_mask = noise_mask * add_noise
        shift_backward = torch.rand(size=durs.shape, device=durs.device) <= 0.5
        shift_backward = shift_backward * noise_mask
        shift_forward = ~shift_backward * noise_mask

        min_end_indices = torch.nn.functional.pad(end_indices[:, :-1], pad=[1, 0]) + 1
        end_indices_noise = end_indices + shift_backward.int()
        end_indices_noise = torch.maximum(end_indices_noise, min_end_indices)
        end_indices_noise = torch.where(noise_mask, end_indices_noise, end_indices)

        max_end_indices = torch.nn.functional.pad(end_indices_noise[:, 1:] - 1, pad=[0, 1])
        end_indices_noise = end_indices_noise + shift_forward.int()
        end_indices_noise = torch.minimum(end_indices_noise, max_end_indices)
        end_indices_noise = torch.where(noise_mask, end_indices_noise, end_indices)

        end_indices_noise = end_indices_noise.int()

        durs_noise = end_indices_noise - torch.nn.functional.pad(end_indices_noise[:, :-1], pad=[1, 0])
        durs_noise = durs_noise * mask

        return durs_noise

    def _sample_lens(self, batch_size, audio_lens, random_sample, max_len=None):
        if max_len is None:
            max_len = self.context_max_len

        # [B]
        max_lens = torch.clamp_max(audio_lens, max=max_len)
        min_lens = torch.clamp_max(max_lens, max=self.context_min_len)

        sample_len_list = []
        for i in range(batch_size):
            if random_sample:
                # Sample between minimum and maximum sample size
                sample_len = torch.randint(low=min_lens[i], high=max_lens[i] + 1, size=[]).item()
            else:
                # Use maximum sample size
                sample_len = max_lens[i].item()

            sample_len_list.append(sample_len)

        sample_lens = torch.tensor(sample_len_list, device=audio_lens.device)
        return sample_lens

    def _find_space_endings(self, text, durs, max_audio_len):
        # [B, T_text]
        cum_ends = torch.cumsum(durs, dim=1).long()
        space_ends = torch.where(text == self.space_token, cum_ends, torch.zeros_like(cum_ends))
        space_ends_invert = torch.where(
            text == self.space_token,
            cum_ends,
            max_audio_len * torch.ones_like(cum_ends)
        )
        # [B, 1]
        if self.pad_with_space:
            min_space = space_ends_invert.topk(k=2, dim=1, largest=False).values[:, 1:2]
            max_space = space_ends.topk(k=2, dim=1).values[:, 1:2]
        else:
            min_space = space_ends_invert.topk(k=1, dim=1, largest=False).values[:, :1]
            max_space = space_ends.topk(k=1, dim=1).values[:, :1]

        return space_ends, min_space, max_space

    def _slice_context_information(
        self,
        text,
        durs,
        audio_tokens,
        audio_codes,
        batch_size,
        context_starts,
        context_ends,
        context_lens,
        text_starts,
        text_ends,
        target_text_lens,
        audio_starts,
        audio_ends,
        target_audio_lens,
    ):
        context_list = []
        target_text_list = []
        target_dur_list = []
        target_audio_token_list = []
        target_audio_code_list = []
        for i in range(batch_size):
            context_start_i = context_starts[i].item()
            context_end_i = context_ends[i].item()
            context_len_i = context_lens[i].item()
            text_start_i = text_starts[i].item()
            text_end_i = text_ends[i].item()
            audio_start_i = audio_starts[i].item()
            audio_end_i = audio_ends[i].item()

            context_i = audio_codes[i, :, context_start_i:context_end_i]
            if context_i.shape[1] < self.context_max_len:
                num_repeats = int(math.ceil(self.context_max_len / context_len_i))
                context_i = torch.tile(context_i, dims=(1, num_repeats))
                context_i = context_i[:, :self.context_max_len]

            text_i = text[i, text_start_i:text_end_i]
            dur_i = durs[i, text_start_i:text_end_i]

            audio_tokens_i = audio_tokens[i, :, audio_start_i:audio_end_i]
            audio_codes_i = audio_codes[i, :, audio_start_i:audio_end_i]

            context_list.append(context_i)
            target_text_list.append(text_i)
            target_dur_list.append(dur_i)
            target_audio_token_list.append(audio_tokens_i)
            target_audio_code_list.append(audio_codes_i)

        context_codes = stack_tensors(tensors=context_list, max_lens=[self.context_max_len]).to(audio_codes.device)
        context_output_lens = self.context_max_len * torch.ones_like(context_lens)

        max_text_len = target_text_lens.max().item()
        target_text = stack_tensors(tensors=target_text_list, max_lens=[max_text_len]).to(audio_codes.device)
        target_durs = stack_tensors(tensors=target_dur_list, max_lens=[max_text_len]).to(audio_codes.device)

        max_audio_len = target_audio_lens.max().item()
        target_audio_tokens = stack_tensors(tensors=target_audio_token_list, max_lens=[max_audio_len]).to(
            audio_codes.device)
        target_audio_codes = stack_tensors(tensors=target_audio_code_list, max_lens=[max_audio_len]).to(
            audio_codes.device)

        return context_codes, context_output_lens, target_text, target_durs, target_audio_tokens, target_audio_codes

    def sample_context_audio_start(self, audio_tokens, audio_codes, audio_lens, text, durs, text_lens, random_sample, max_len=None):
        batch_size = audio_codes.shape[0]
        max_audio_len = audio_tokens.shape[2]
        # [B, 1]
        context_ends = self._sample_lens(batch_size=batch_size, audio_lens=audio_lens, random_sample=random_sample, max_len=max_len)
        context_ends = rearrange(context_ends, 'B -> B 1')
        space_ends, min_space, max_space = self._find_space_endings(text=text, durs=durs, max_audio_len=max_audio_len)

        # [B, T]
        valid_space = torch.logical_and(space_ends >= min_space, space_ends <= max_space)
        valid_space = torch.logical_and(valid_space, space_ends <= context_ends)
        valid_space = torch.logical_or(valid_space, space_ends == min_space)
        space_ends = torch.where(valid_space, space_ends, torch.zeros_like(space_ends))

        # [B]
        context_end_topk = space_ends.topk(k=1, dim=1)
        context_lens = context_end_topk.values[:, 0]
        target_audio_lens = audio_lens - context_lens

        target_text_starts = context_end_topk.indices[:, 0] + torch.tensor(1)
        target_text_lens = text_lens - target_text_starts

        audio_starts = context_lens
        target_audio_lens = torch.maximum(target_audio_lens, target_text_lens)

        if self.context_len_noise and random_sample:
            min_context_len = torch.clamp_max(input=context_lens, max=self.context_min_len)
            context_len_noise = torch.randint_like(input=context_lens, low=0, high=self.context_len_noise + 1)
            context_lens = context_lens - context_len_noise
            context_lens = torch.maximum(input=context_lens, other=min_context_len)

        zero_starts = torch.zeros(batch_size, dtype=torch.int32)
        context_codes, context_lens, target_text, target_durs, target_audio_tokens, target_audio_codes = \
            self._slice_context_information(
                text=text,
                durs=durs,
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                batch_size=batch_size,
                context_starts=zero_starts,
                context_ends=context_lens,
                context_lens=context_lens,
                text_starts=target_text_starts,
                text_ends=text_lens,
                target_text_lens=target_text_lens,
                audio_starts=audio_starts,
                audio_ends=audio_lens,
                target_audio_lens=target_audio_lens,
            )

        return target_text, target_durs, target_text_lens, target_audio_tokens, target_audio_codes, target_audio_lens, \
               context_codes, context_lens

    def sample_context_audio_end(self, audio_tokens, audio_codes, audio_lens, text, durs):
        batch_size = audio_codes.shape[0]
        max_audio_len = audio_tokens.shape[2]
        # [B, 1]
        context_rand_lens = self._sample_lens(batch_size=batch_size, audio_lens=audio_lens, random_sample=True)
        context_starts = audio_lens - context_rand_lens
        context_starts = rearrange(context_starts, 'B -> B 1')
        space_ends, min_space, max_space = self._find_space_endings(text=text, durs=durs, max_audio_len=max_audio_len)

        # [B, T]
        valid_space = torch.logical_and(space_ends >= min_space, space_ends <= max_space)
        valid_space = torch.logical_and(valid_space, space_ends >= context_starts)
        valid_space = torch.logical_or(valid_space, space_ends == max_space)
        space_ends = torch.where(valid_space, space_ends, max_audio_len * torch.ones_like(space_ends))

        # [B]
        context_start_topk = space_ends.topk(k=1, dim=1, largest=False)
        target_audio_lens = context_start_topk.values[:, 0]
        context_lens = audio_lens - target_audio_lens
        target_text_lens = context_start_topk.indices[:, 0] + torch.tensor(1)

        target_audio_lens = torch.maximum(target_audio_lens, target_text_lens)

        if self.context_len_noise:
            min_context_len = torch.clamp_max(input=context_lens, max=self.context_min_len)
            context_len_noise = torch.randint_like(input=context_lens, low=0, high=self.context_len_noise + 1)
            context_lens = context_lens - context_len_noise
            context_lens = torch.maximum(input=context_lens, other=min_context_len)
            context_starts = audio_lens - context_lens

        zero_starts = torch.zeros(batch_size, dtype=torch.int32)
        context_codes, context_lens, target_text, target_durs, target_audio_tokens, target_audio_codes = \
            self._slice_context_information(
                text=text,
                durs=durs,
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                batch_size=batch_size,
                context_starts=context_starts,
                context_ends=audio_lens,
                context_lens=context_lens,
                text_starts=zero_starts,
                text_ends=target_text_lens,
                target_text_lens=target_text_lens,
                audio_starts=zero_starts,
                audio_ends=target_audio_lens,
                target_audio_lens=target_audio_lens,
            )

        return target_text, target_durs, target_text_lens, target_audio_tokens, target_audio_codes, target_audio_lens, \
               context_codes, context_lens

    def _concat_tensors(self, inputs1, inputs2, pad_value=0.0):
        len1 = inputs1.shape[-1]
        len2 = inputs2.shape[-1]
        padding1 = max(len2 - len1, 0)
        padding2 = max(len1 - len2, 0)
        out1 = torch.nn.functional.pad(inputs1, pad=[0, padding1], value=pad_value)
        out2 = torch.nn.functional.pad(inputs2, pad=[0, padding2], value=pad_value)
        out = torch.cat([out1, out2], dim=0)
        return out

    def sample_context_audio_batch(self, audio_tokens, audio_codes, audio_lens, text, text_lens, durs):
        half_batch_size = audio_tokens.shape[0] // 2
        text_sample1, dur_sample1, text_sample_lens1, audio_token_sample1, \
        audio_codes_sample1, audio_token_sample_lens1, context_codes1, context_lens1 = self.sample_context_audio_start(
            audio_tokens=audio_tokens[:half_batch_size],
            audio_codes=audio_codes[:half_batch_size],
            audio_lens=audio_lens[:half_batch_size],
            text=text[:half_batch_size],
            text_lens=text_lens[:half_batch_size],
            durs=durs[:half_batch_size],
            random_sample=True
        )
        text_sample2, dur_sample2, text_sample_lens2, audio_token_sample2, \
        audio_codes_sample2, audio_token_sample_lens2, context_codes2, context_lens2 = self.sample_context_audio_end(
            audio_tokens=audio_tokens[half_batch_size:],
            audio_codes=audio_codes[half_batch_size:],
            audio_lens=audio_lens[half_batch_size:],
            text=text[half_batch_size:],
            durs=durs[half_batch_size:],
        )
        text_sample = self._concat_tensors(text_sample1, text_sample2, pad_value=self.text_pad_token)
        dur_sample = self._concat_tensors(dur_sample1, dur_sample2)
        text_sample_lens = torch.cat([text_sample_lens1, text_sample_lens2], dim=0)
        audio_token_sample = self._concat_tensors(audio_token_sample1, audio_token_sample2)
        audio_codes_sample = self._concat_tensors(audio_codes_sample1, audio_codes_sample2)
        audio_token_sample_lens = torch.cat([audio_token_sample_lens1, audio_token_sample_lens2], dim=0)
        context_codes = self._concat_tensors(context_codes1, context_codes2)
        context_lens = torch.cat([context_lens1, context_lens2], dim=0)

        return text_sample, dur_sample, text_sample_lens, audio_token_sample, \
               audio_codes_sample, audio_token_sample_lens, context_codes, context_lens

    def get_context(self, audio_tokens, audio_lens, text, text_lens, max_len=None):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_lens)

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, _, _ = self.aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_lens,
        )

        _, _, _, _, _, _, \
        context_codes, context_lens = self.sample_context_audio_start(
            audio_tokens=audio_tokens,
            audio_codes=audio_codes,
            audio_lens=audio_lens,
            text=text,
            text_lens=text_lens,
            durs=durs,
            random_sample=False,
            max_len=max_len,
        )
        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context_emb, context, context_lens

    def get_context_audio(self, audio_tokens, audio_lens):
        batch_size = audio_tokens.shape[0]
        context_lens = torch.clamp_max(audio_lens, max=self.context_max_len)
        context_token_list = []
        for i in range(batch_size):
            context_len_i = context_lens[i]
            context_tokens_i = audio_tokens[i, :, :context_len_i]
            context_token_list.append(context_tokens_i)

        max_context_len = max(context_lens)
        context_tokens = stack_tensors(tensors=context_token_list, max_lens=[max_context_len]).to(audio_tokens.device)

        context_tokens_rearrange = rearrange(context_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        context_codes = self.vector_quantizer.decode(indices=context_tokens_rearrange, input_len=context_lens)

        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context_emb, context, context_lens

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "sample_context": NeuralType((), BoolType(), optional=True),
        },
        output_types={
            "semantic_token_sample": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_sample_lens": NeuralType(tuple('B'), LengthsType()),
            "semantic_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "semantic_tokens_pred_post": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits_post": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "semantic_loss_mask": NeuralType(('B', 'T_text'), MaskType()),
            "speaking_rate_indices": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_indices_pred": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_logits": NeuralType(('B', 'C'), LogitsType()),
            "text_sample_lens": NeuralType(tuple('B'), LengthsType()),
            "align_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
            "dalign_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "dalign_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "dalign_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        }
    )
    def forward(
        self,
        text,
        text_lens,
        audio_tokens,
        audio_token_lens,
    ):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens).detach()

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, align_hard, align_soft, align_logits = self.aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
        )

        if self.training:
            text_sample, dur_sample, text_sample_lens, audio_token_sample, \
            audio_codes_sample, audio_token_sample_lens, context_codes, context_lens = self.sample_context_audio_batch(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                durs=durs,
            )
            semantic_maskin, semantic_loss_mask = self.create_infill_mask(
                input_lens=audio_token_sample_lens,
                dist=self.semantic_infill_dist,
                infill_min=self.semantic_infill_min,
                infill_max=self.semantic_infill_max,
            )
        else:
            text_sample, dur_sample, text_sample_lens, audio_token_sample, \
            audio_codes_sample, audio_token_sample_lens, context_codes, context_lens = self.sample_context_audio_start(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                durs=durs,
                random_sample=False
            )

            audio_mask = get_mask_from_lengths(audio_token_sample_lens)
            semantic_maskin = torch.zeros(
                [audio_codes_sample.shape[0], audio_codes_sample.shape[2]], dtype=torch.bool, device=audio_tokens.device
            )
            # Unmask every 10th element
            semantic_maskin[:, ::10] = True
            semantic_maskin = semantic_maskin * audio_mask
            semantic_loss_mask = ~semantic_maskin * audio_mask

        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        _, speaking_rate_indices = self.get_speaking_rate(text_lens=text_lens, durs=durs)

        semantic_token_sample = audio_token_sample[:, :self.semantic_codebook_num, :]
        semantic_codes = audio_codes_sample[:, :self.semantic_codebook_dim, :]

        semantic_tokens_pred, semantic_token_logits, semantic_tokens_pred_post, semantic_token_logits_post, \
        dalign_hard, dalign_soft, dalign_logits, \
        speaking_rate_indices_pred, speaking_rate_logits = self.forward_internal(
            text=text_sample,
            text_lens=text_sample_lens,
            context_emb=context_emb,
            context=context,
            context_lens=context_lens,
            semantic_codes=semantic_codes,
            audio_lens=audio_token_sample_lens,
            semantic_maskin=semantic_maskin,
            durs=dur_sample,
        )

        return (
            semantic_token_sample,
            audio_token_sample_lens,
            semantic_tokens_pred,
            semantic_token_logits,
            semantic_tokens_pred_post,
            semantic_token_logits_post,
            semantic_loss_mask,
            speaking_rate_indices,
            speaking_rate_indices_pred,
            speaking_rate_logits,
            text_sample_lens,
            align_hard,
            align_soft,
            align_logits,
            dalign_hard,
            dalign_soft,
            dalign_logits,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "num_audio_iters": NeuralType((), IntType()),
            "audio_temperature": NeuralType((), FloatType(), optional=True),
            "audio_topk": NeuralType((), IntType(), optional=True),
        },
        output_types={
            "semantic_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "dur_lens": NeuralType(tuple('B'), LengthsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "balign_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
        }
    )
    def infer_gta(
        self,
        text,
        text_lens,
        audio_tokens,
        audio_token_lens,
        num_audio_iters=1,
        audio_temperature=None,
        audio_topk=None,
    ):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T').detach()
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens)

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, align_soft, _ = self.aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
        )

        _, _, _, _, _, _, context_codes, context_lens = self.sample_context_audio_start(
            audio_tokens=audio_tokens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            text=text,
            text_lens=text_lens,
            durs=durs,
            random_sample=False
        )
        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )

        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)
        context = rearrange(context, 'B D T -> B T D')
        # [batch_size, text_len, hidden_dim]
        text_enc = self.text_encoder(
            text=text, text_lens=text_lens, audio_lens=audio_token_lens, context_emb=context_emb, context=context, context_mask=context_mask
        )

        text_enc_align = rearrange(text_enc, 'B T D -> B D T')
        durs, _, dalign_soft, _ = self.decoder_aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=text_enc_align,
            audio_lens=audio_token_lens,
        )

        audio_mask = get_mask_from_lengths(audio_token_lens)
        if self.text_conditionining_layer is not None:
            text_enc = self.text_conditionining_layer(inputs=text_enc, text=text, durations=durs, audio_mask=audio_mask)

        semantic_enc = self.encoder(
            inputs=text_enc, audio_mask=audio_mask, context=context, context_mask=context_mask
        )
        # [B, C_semantic, T]
        semantic_tokens = self._semantic_token_infer(
            inputs=semantic_enc,
            audio_lens=audio_token_lens,
            context=context,
            context_mask=context_mask,
            num_iters=num_audio_iters,
            temperature=audio_temperature,
            topk=audio_topk,
        )

        return semantic_tokens, align_soft, dalign_soft

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            semantic_token_sample,
            audio_token_sample_lens,
            _,
            semantic_token_logits,
            _,
            semantic_token_logits_post,
            semantic_loss_mask,
            speaking_rate_indices,
            _,
            speaking_rate_logits,
            text_sample_lens,
            align_hard,
            align_soft,
            align_logits,
            dalign_hard,
            dalign_soft,
            dalign_logits,
        ) = self(
            text=text,
            text_lens=text_lens,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
        )

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)

        semantic_token_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits, target_tokens=semantic_token_sample, mask=audio_mask
        )
        train_semantic_token_loss = self.audio_token_loss_scale * semantic_token_loss

        semantic_token_post_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits_post, target_tokens=semantic_token_sample, mask=semantic_loss_mask
        )
        train_semantic_token_post_loss = self.audio_token_loss_scale * semantic_token_post_loss

        speaking_rate_loss = self.speaking_rate_loss_fn(logits=speaking_rate_logits, target_index=speaking_rate_indices.detach())
        train_speaking_rate_loss = self.speaking_rate_loss_scale * speaking_rate_loss

        ctc_loss = self.forward_sum_loss_fn(attn_logprob=align_logits, in_lens=text_lens, out_lens=audio_token_lens)
        train_ctc_loss = self.aligner_ctc_loss_scale * ctc_loss

        ctc_decoder_loss = self.forward_sum_loss_fn(
            attn_logprob=dalign_logits, in_lens=text_sample_lens, out_lens=audio_token_sample_lens
        )
        train_ctc_decoder_loss = self.aligner_ctc_loss_scale * ctc_decoder_loss

        if self.current_epoch < self.bin_loss_start_epoch:
            bin_loss_weight = 0.0
        elif self.current_epoch >= self.bin_loss_warmup_epochs:
            bin_loss_weight = 1.0
        else:
            bin_loss_weight = (self.current_epoch - self.bin_loss_start_epoch) / (self.bin_loss_warmup_epochs - self.bin_loss_start_epoch)

        bin_loss = self.bin_loss_fn(hard_attention=align_hard, soft_attention=align_soft)
        train_bin_loss = bin_loss_weight * self.aligner_bin_loss_scale * bin_loss

        bin_decoder_loss = self.bin_loss_fn(hard_attention=dalign_hard, soft_attention=dalign_soft)
        train_bin_decoder_loss = bin_loss_weight * self.aligner_bin_loss_scale * bin_decoder_loss

        loss = train_semantic_token_loss + train_semantic_token_post_loss + \
               train_speaking_rate_loss + train_ctc_loss + train_bin_loss + \
               train_ctc_decoder_loss + train_bin_decoder_loss

        metrics = {
            "t_semantic_token_loss": semantic_token_loss,
            "t_semantic_token_post_loss": semantic_token_post_loss,
            "t_speaking_rate_loss": speaking_rate_loss,
            "t_ctc_loss": ctc_loss,
            "t_ctc_decoder_loss": ctc_decoder_loss,
            "t_bin_loss": bin_loss,
            "t_bin_decoder_loss": bin_decoder_loss,
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", semantic_token_loss, prog_bar=True, logger=False, sync_dist=True)

        return loss

    def validation_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            semantic_token_sample,
            audio_token_sample_lens,
            semantic_tokens_pred,
            semantic_token_logits,
            semantic_tokens_pred_post,
            semantic_token_logits_post,
            semantic_loss_mask,
            speaking_rate_indices,
            speaking_rate_indices_pred,
            speaking_rate_logits,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self(
            text=text,
            text_lens=text_lens,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
        )

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)

        semantic_token_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits, target_tokens=semantic_token_sample, mask=audio_mask
        )
        semantic_token_correct = (semantic_token_sample == semantic_tokens_pred) * rearrange(audio_mask, 'B T -> B 1 T')
        semantic_token_accuracy = semantic_token_correct.sum() / audio_token_sample_lens.sum() / self.semantic_codebook_num

        semantic_token_post_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits_post, target_tokens=semantic_token_sample, mask=semantic_loss_mask
        )
        semantic_token_correct_post = (semantic_token_sample == semantic_tokens_pred_post) * rearrange(semantic_loss_mask, 'B T -> B 1 T')
        semantic_token_post_accuracy = semantic_token_correct_post.sum() / semantic_loss_mask.sum() / self.semantic_codebook_num

        speaking_rate_loss = self.speaking_rate_loss_fn(logits=speaking_rate_logits, target_index=speaking_rate_indices)
        speaking_rate_correct = (speaking_rate_indices == speaking_rate_indices_pred)
        speaking_rate_accuracy = speaking_rate_correct.float().mean()

        metrics = {
            "val_loss": semantic_token_loss,
            "val_semantic_token_loss": semantic_token_loss,
            "val_semantic_token_accuracy": semantic_token_accuracy,
            "val_semantic_token_post_loss": semantic_token_post_loss,
            "val_semantic_token_post_accuracy": semantic_token_post_accuracy,
            "val_speaking_rate_loss": speaking_rate_loss,
            "val_speaking_rate_accuracy": speaking_rate_accuracy,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def _setup_train_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=True
        )

        sampler = dataset.get_sampler(dataloader_params.batch_size, world_size=self.trainer.world_size)
        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **dataloader_params
        )

    def _setup_test_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=False,
            phoneme_probability=self.inference_phoneme_probability,
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def configure_callbacks(self):
        if not self.log_config:
            return []

        data_loader = self._setup_test_dataloader(
            dataset_config=self.log_config.dataset, dataloader_params=self.log_config.dataloader_params
        )
        generators = instantiate(self.log_config.generators)
        log_dir = Path(self.log_config.log_dir) if self.log_config.log_dir else None
        log_callback = LoggingCallback(
            generators=generators,
            data_loader=data_loader,
            log_epochs=self.log_config.log_epochs,
            epoch_frequency=self.log_config.epoch_frequency,
            output_dir=log_dir,
            loggers=self.trainer.loggers,
            log_tensorboard=self.log_config.log_tensorboard,
            log_wandb=self.log_config.log_wandb,
            max_filename_len=self.log_config.max_filename_len,
        )

        return [log_callback]

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []

    def _semantic_token_infer(
        self,
        inputs,
        audio_lens,
        context,
        context_mask,
        num_iters,
        temperature=None,
        topk=None,
    ):
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens)
        num_tokens = inputs.shape[1]

        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        # [B, T]
        audio_maskin = torch.zeros_like(audio_mask, dtype=torch.bool)
        # [B, T, C]
        audio_token_shape = [audio_mask.shape[0], audio_mask.shape[1], self.decoder.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [audio_mask.shape[0], audio_mask.shape[1], self.decoder.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        for i in range(num_iters):
            if i == 0:
                # [B, C, T], [B, C, W, T]
                audio_tokens_i, audio_logits = self.decoder.forward_parallel(
                    inputs=inputs, audio_mask=audio_mask, temperature=temperature, topk=topk,
                )
            else:
                # [B, C, T], [B, C, W, T]
                audio_tokens_i, audio_logits = self.decoder(
                    inputs=inputs,
                    audio_mask=audio_mask,
                    context=context,
                    context_mask=context_mask,
                    audio_codes=audio_codes,
                    audio_maskin=audio_maskin,
                    temperature=temperature,
                    topk=topk,
                )
            audio_tokens_i = rearrange(audio_tokens_i, 'B C T -> B T C')
            audio_tokens_rearrange_i = rearrange(audio_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            audio_codes_i = self.vector_quantizer_semantic.decode(indices=audio_tokens_rearrange_i, input_len=audio_lens)
            audio_codes_i = rearrange(audio_codes_i, 'B D T -> B T D')

            top_i = torch.clamp_max(index_shift + i, max=num_tokens - 1)
            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)
            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(audio_mask, maskin_i, False)
            maskin_i = torch.where(audio_maskin, False, maskin_i)
            maskin_3d_i = rearrange(maskin_i, 'B T -> B T 1')

            audio_tokens = torch.where(maskin_3d_i, audio_tokens_i, audio_tokens)
            audio_codes = torch.where(maskin_3d_i, audio_codes_i, audio_codes)

            audio_maskin = torch.logical_or(audio_maskin, maskin_i)

        audio_maskin_3d = rearrange(audio_maskin, 'B T -> B T 1')
        audio_tokens = torch.where(audio_maskin_3d, audio_tokens, audio_tokens_i)

        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens

    def _duration_infer(
        self, inputs, text_lens, context, context_mask, num_iters, temperature=None, topk=None
    ):
        # [B, T]
        text_mask = get_mask_from_lengths(text_lens)

        num_tokens = inputs.shape[1]
        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        duration_maskin = torch.zeros_like(text_mask, dtype=torch.bool)
        dur_indices = torch.zeros_like(text_mask, dtype=torch.int)

        for i in range(num_iters):
            if i == 0:
                # [B, C, T], [B, C, W, T]
                dur_indices_i, dur_logits = self.duration_decoder.forward_parallel(
                    inputs=inputs, text_mask=text_mask, temperature=temperature, topk=topk
                )
            else:
                dur_indices_i, dur_logits = self.duration_decoder(
                    inputs=inputs,
                    dur_indices=dur_indices,
                    text_mask=text_mask,
                    duration_maskin=duration_maskin,
                    context=context,
                    context_mask=context_mask,
                    temperature=temperature,
                    topk=topk
                )

            top_i = torch.clamp_max(index_shift + 1, max=num_tokens - 1)

            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)
            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(text_mask, maskin_i, False)
            maskin_i = torch.where(duration_maskin, False, maskin_i)

            dur_indices = torch.where(maskin_i, dur_indices_i, dur_indices)
            duration_maskin = torch.logical_or(duration_maskin, maskin_i)

        dur_indices = torch.where(duration_maskin, dur_indices, dur_indices_i)
        # [B, T]
        durs = self.index_to_duration(dur_indices=dur_indices, mask=text_mask)

        return durs, dur_indices

    @typecheck(
        input_types={
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "lengths": NeuralType(tuple('B'), LengthsType())
        },
        output_types={
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex())
        }
    )
    def duration_to_index(self, durs, lengths):
        durs = torch.clamp(durs.float(), min=1, max=self.max_token_duration)
        dur_indices = durs - 1
        dur_indices = mask_sequence_tensor(tensor=dur_indices, lengths=lengths)
        dur_indices = dur_indices.int()
        return dur_indices

    def index_to_duration(self, dur_indices, mask):
        # [B, T]
        durs = dur_indices + 1
        durs = durs * mask
        return durs

    @typecheck(
        input_types={
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
        },
        output_types={
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
            "speaking_rate_indices": NeuralType(tuple('B'), TokenIndex()),
        }
    )
    def get_speaking_rate(self, text_lens, durs):
        sr_text_len = torch.clamp_min(text_lens - 2, min=1)

        text_mask = get_mask_from_lengths(text_lens)
        max_text_len = text_mask.shape[1]
        indices = torch.arange(max_text_len, device=durs.device) + 1
        dur_mask = torch.where(
            rearrange(indices, 'T -> 1 T') == rearrange(text_lens, 'B -> B 1'),
            torch.zeros_like(text_mask),
            text_mask
        )
        sr_durs = torch.clamp(durs.float(), min=1, max=self.max_token_duration)
        sr_durs = sr_durs * dur_mask
        sr_durs = sr_durs[:, 1:]
        sr_audio_lens = sr_durs.sum(dim=1)

        fps = (-1.0) * sr_audio_lens / sr_text_len.float()
        speaking_rate, speaking_rate_indices = self.speaking_rate_quantizer(inputs=fps)

        return speaking_rate, speaking_rate_indices

    def get_audio_lens(self, speaking_rate, text_lens):
        audio_lens = (-1.0) * speaking_rate * text_lens
        audio_lens = torch.round(audio_lens).int()
        audio_lens = torch.max(audio_lens, text_lens)
        return audio_lens

    def forward_internal(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        semantic_codes,
        audio_lens,
        semantic_maskin,
        durs,
    ):
        audio_mask = get_mask_from_lengths(audio_lens)
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')
        speaking_rate_indices_pred, speaking_rate_logits = self.speaking_rate_predictor(context_emb=context_emb)
        # [batch_size, text_len, hidden_dim]
        text_enc = self.text_encoder(
            text=text, text_lens=text_lens, audio_lens=audio_lens, context_emb=context_emb, context=context, context_mask=context_mask
        )

        text_enc_align = rearrange(text_enc, 'B T D -> B D T')
        _, dalign_hard, dalign_soft, dalign_logits = self.decoder_aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=text_enc_align,
            audio_lens=audio_lens,
        )

        if self.text_conditionining_layer is not None:
            if self.training:
                durs = self._add_decoder_duration_noise(durs=durs, text_lens=text_lens)
            text_enc = self.text_conditionining_layer(inputs=text_enc, text=text, durations=durs, audio_mask=audio_mask)

        semantic_enc = self.encoder(
            inputs=text_enc, audio_mask=audio_mask, context=context, context_mask=context_mask
        )
        semantic_tokens_pred, semantic_logits = self.decoder.forward_parallel(inputs=semantic_enc, audio_mask=audio_mask)

        semantic_codes = rearrange(semantic_codes, 'B C T -> B T C')
        semantic_tokens_pred_post, semantic_logits_post = self.decoder(
            inputs=semantic_enc,
            audio_mask=audio_mask,
            context=context,
            context_mask=context_mask,
            audio_codes=semantic_codes,
            audio_maskin=semantic_maskin,
        )

        return semantic_tokens_pred, semantic_logits, semantic_tokens_pred_post, semantic_logits_post, \
               dalign_hard, dalign_soft, dalign_logits, \
               speaking_rate_indices_pred, speaking_rate_logits,

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "num_audio_iters": NeuralType((), IntType(), optional=True),
            "audio_topk": NeuralType((), IntType(), optional=True),
            "audio_temperature": NeuralType((), FloatType(), optional=True),
            "num_duration_iters": NeuralType((), IntType(), optional=True),
            "duration_topk": NeuralType((), IntType(), optional=True),
            "duration_temperature": NeuralType((), FloatType(), optional=True),
            "speaking_rate": NeuralType(tuple('B'), FloatType(), optional=True),
            "silence_pad_start": NeuralType((), IntType(), optional=True),
            "silence_pad_end": NeuralType((), IntType(), optional=True),
            "min_speaking_rate": NeuralType((), IntType(), optional=True),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_token'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType())
        }
    )
    def infer(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        num_audio_iters=1,
        audio_topk=None,
        audio_temperature=None,
        speaking_rate=None,
        silence_pad_start=5,
        silence_pad_end=10,
        min_speaking_rate=-0.5,
        max_speaking_rate=0.5,
    ):
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')

        if speaking_rate is None:
            speaking_rate_indices, _ = self.speaking_rate_predictor(context_emb=context_emb)
            speaking_rate = self.speaking_rate_quantizer.get_codes(indices=speaking_rate_indices)
            speaking_rate = torch.clamp(speaking_rate, min=min_speaking_rate, max=max_speaking_rate)

        audio_lens = self.get_audio_lens(speaking_rate=speaking_rate, text_lens=text_lens)
        audio_mask = get_mask_from_lengths(audio_lens)

        # [batch_size, text_len, hidden_dim]
        text_enc = self.text_encoder(
            text=text, text_lens=text_lens, audio_lens=audio_lens, context_emb=context_emb, context=context, context_mask=context_mask
        )
        if self.text_conditionining_layer is not None:
            text_enc_align = rearrange(text_enc, 'B T D -> B D T')
            durs, _, _, _ = self.decoder_aligner(
                text=text,
                text_lens=text_lens,
                audio_codes=text_enc_align,
                audio_lens=audio_lens,
            )
            text_enc = self.text_conditionining_layer(inputs=text_enc, text=text, durations=durs, audio_mask=audio_mask)

            #if silence_pad_start:
            #    for i in range(durs.shape[0]):
            #        durs[i, 0] = silence_pad_start

            #if silence_pad_end:
            #    for i in range(durs.shape[0]):
            #        durs[i, dur_lens[i] - 1] = silence_pad_end


        semantic_enc = self.encoder(
            inputs=text_enc, audio_mask=audio_mask, context=context, context_mask=context_mask
        )
        # [B, C_semantic, T]
        semantic_tokens = self._semantic_token_infer(
            inputs=semantic_enc,
            audio_lens=audio_lens,
            context=context,
            context_mask=context_mask,
            num_iters=num_audio_iters,
            temperature=audio_temperature,
            topk=audio_topk,
        )

        return semantic_tokens, audio_lens