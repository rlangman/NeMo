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

from collections import defaultdict
from pathlib import Path
import logging
import math
import numpy as np
from typing import List

import torch
from einops import rearrange
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.discrete_speech_loss import AudioTokenLoss, MaskedSoftmax, SpeakingRateLoss
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, regulate_len, stack_tensors
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    LogprobsType,
    ProbsType,
    TokenDurationType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import model_utils
from nemo.utils.decorators import experimental


@experimental
class DiscreteSpeechModel(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = self._create_tokenizer(cfg.text_tokenizer)
        self.pad_with_space = self.text_tokenizer.pad_with_space
        self.inference_phoneme_probability = cfg.get("inference_phoneme_probability", 0.0)

        super().__init__(cfg=cfg, trainer=trainer)

        # Text tokenizer information
        num_text_embed = len(self.text_tokenizer.tokens)
        self.text_pad_token = self.text_tokenizer.pad
        self.space_token = self.text_tokenizer.space
        if self.text_tokenizer.add_boundaries:
            self.bos_token = self.text_tokenizer.bos
            self.eos_token = self.text_tokenizer.eos
        else:
            self.bos_token = None
            self.eos_token = None

        # Maximum duration of a single multiphone
        self.max_token_duration = cfg.get("max_token_duration")

        self.context_min_len = cfg.get("context_min_len", 75)
        # Context length in terms of number of audio tokens
        self.context_max_len = cfg.get("context_max_len", 250)

        # Quantizer definitions
        self.semantic_codebook_num = cfg.get("semantic_codebook_num")
        self.acoustic_codebook_num = cfg.get("acoustic_codebook_num")
        self.vector_quantizer = instantiate(cfg.vector_quantizer)

        self.speaking_rate_quantizer = instantiate(cfg.speaking_rate_quantizer)

        self.text_down_sample_rate = cfg.get("text_down_sample_rate", 1)
        self.space_dur = cfg.get("space_dur", 1)

        # Encoder, decoder definitions
        self.text_encoder = instantiate(
            cfg.text_encoder,
            n_embed=num_text_embed,
            padding_idx=self.text_pad_token,
            down_sample_rate=self.text_down_sample_rate,
            bos_id=self.bos_token,
            eos_id=self.eos_token,
            space_id=self.space_token,
            space_dur=self.space_dur,
        )
        self.decoder = instantiate(cfg.decoder)
        self.duration_decoder = instantiate(cfg.duration_decoder)
        self.speaking_rate_predictor = instantiate(cfg.speaking_rate_predictor)

        # Context encoder definition
        self.context_encoder = instantiate(cfg.context_encoder)

        if self.text_down_sample_rate == 1:
            self.aligner = instantiate(cfg.aligner, num_text_emb=num_text_embed)
        elif self.text_down_sample_rate > 1:
            self.aligner = instantiate(
                alignment_encoder=cfg.aligner,
                num_text_emb=num_text_embed,
                down_sample_rate=self.text_down_sample_rate,
                bos_id=self.bos_token,
                eos_id=self.eos_token,
                space_id=self.space_token,
                space_dur=self.space_dur,
            )
        else:
            raise ValueError(f"text_down_sample_rate must be >= 1")

        self.freeze_aligner = cfg.get("freeze_aligner", False)
        if self.freeze_aligner:
            self.aligner.eval()
            self.aligner.freeze()

        self.duration_infill_min = cfg.get("duration_infill_min", 0.25)
        self.duration_infill_max = cfg.get("duration_infill_max", 1.0)
        duration_infill_beta = cfg.get("duration_infill_beta", 2.0)
        self.duration_infill_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=duration_infill_beta)

        # Reconstruction losses
        self.audio_token_loss_scale = cfg.get("audio_token_loss_scale", 1.0)
        audio_label_smoothing = cfg.get("audio_label_smoothing", 0.0)
        self.semantic_token_loss_fn = AudioTokenLoss(
            num_codebooks=self.semantic_codebook_num, label_smoothing=audio_label_smoothing
        )
        self.acoustic_token_loss_fn = AudioTokenLoss(
            num_codebooks=self.acoustic_codebook_num, label_smoothing=audio_label_smoothing
        )

        self.duration_loss_scale = cfg.get("duration_loss_scale", 1e-3)
        duration_label_smoothing = cfg.get("duration_label_smoothing", 0.0)
        self.duration_loss_fn = MaskedSoftmax(label_smoothing=duration_label_smoothing)

        self.speaking_rate_loss_scale = cfg.get("speaking_rate_loss_scale", 1e-4)
        speaking_rate_label_smoothing = cfg.get("speaking_rate_label_smoothing", 0.0)
        self.speaking_rate_loss_fn = SpeakingRateLoss(label_smoothing=speaking_rate_label_smoothing)

        # Aligner losses
        self.aligner_bin_loss_scale = cfg.get("aligner_bin_loss_scale", 0.01)
        self.aligner_ctc_loss_scale = cfg.get("aligner_ctc_loss_scale", 0.01)
        self.bin_loss_start_epoch = cfg.get("bin_loss_start_epoch", 0)
        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 10)

        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()

        self.skip_nan_gradients = cfg.get("skip_nan_gradients", True)

        self.log_config = cfg.get("log_config", None)

    def _create_tokenizer(self, tokenizer_config):
        if "g2p" in tokenizer_config:
            if "phoneme_dict" in tokenizer_config.g2p:
                tokenizer_config.g2p.phoneme_dict = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict',
                    tokenizer_config.g2p.phoneme_dict,
                )

            if "heteronyms" in tokenizer_config.g2p:
                tokenizer_config.g2p.heteronyms = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms',
                    tokenizer_config.g2p.heteronyms,
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
        space_ends_invert = torch.where(text == self.space_token, cum_ends, max_audio_len * torch.ones_like(cum_ends))
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
                context_i = context_i[:, : self.context_max_len]

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
            audio_codes.device
        )
        target_audio_codes = stack_tensors(tensors=target_audio_code_list, max_lens=[max_audio_len]).to(
            audio_codes.device
        )

        return context_codes, context_output_lens, target_text, target_durs, target_audio_tokens, target_audio_codes

    def get_context(self, audio_tokens, audio_len):
        context_tokens = audio_tokens[:, :, :self.context_max_len]
        context_len = torch.clamp_max(input=audio_len, max=self.context_max_len)

        context_tokens_rearrange = rearrange(context_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        context_codes = self.vector_quantizer.decode(indices=context_tokens_rearrange, input_len=context_len)
        context_emb = self.context_encoder(
            audio_codes=context_codes,
            audio_len=context_len,
        )
        return context_emb

    def sample_context_audio_start(
        self, audio_tokens, audio_codes, audio_lens, text, durs, text_lens, random_sample, max_len=None
    ):
        batch_size = audio_codes.shape[0]
        max_audio_len = audio_tokens.shape[2]
        # [B, 1]
        context_ends = self._sample_lens(
            batch_size=batch_size, audio_lens=audio_lens, random_sample=random_sample, max_len=max_len
        )
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

        zero_starts = torch.zeros(batch_size, dtype=torch.int32)
        context_codes, context_lens, target_text, target_durs, target_audio_tokens, target_audio_codes = (
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
        )

        return (
            target_text,
            target_durs,
            target_text_lens,
            target_audio_tokens,
            target_audio_codes,
            target_audio_lens,
            context_codes,
            context_lens,
        )

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

        zero_starts = torch.zeros(batch_size, dtype=torch.int32)
        context_codes, context_lens, target_text, target_durs, target_audio_tokens, target_audio_codes = (
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
        )

        return (
            target_text,
            target_durs,
            target_text_lens,
            target_audio_tokens,
            target_audio_codes,
            target_audio_lens,
            context_codes,
            context_lens,
        )

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
        (
            text_sample1,
            dur_sample1,
            text_sample_lens1,
            audio_token_sample1,
            audio_codes_sample1,
            audio_token_sample_lens1,
            context_codes1,
            context_lens1,
        ) = self.sample_context_audio_start(
            audio_tokens=audio_tokens[:half_batch_size],
            audio_codes=audio_codes[:half_batch_size],
            audio_lens=audio_lens[:half_batch_size],
            text=text[:half_batch_size],
            text_lens=text_lens[:half_batch_size],
            durs=durs[:half_batch_size],
            random_sample=True,
        )
        (
            text_sample2,
            dur_sample2,
            text_sample_lens2,
            audio_token_sample2,
            audio_codes_sample2,
            audio_token_sample_lens2,
            context_codes2,
            context_lens2,
        ) = self.sample_context_audio_end(
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

        return (
            text_sample,
            dur_sample,
            text_sample_lens,
            audio_token_sample,
            audio_codes_sample,
            audio_token_sample_lens,
            context_codes,
            context_lens,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_len": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "audio_token_sample": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_sample_len": NeuralType(tuple('B'), LengthsType()),
            "semantic_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "semantic_tokens_pred_pre": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits_pre": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "acoustic_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "acoustic_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "text_sample_len": NeuralType(tuple('B'), LengthsType()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'D', 'T_text'), LogitsType()),
            "dur_indices_pred_pre": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits_pre": NeuralType(('B', 'D', 'T_text'), LogitsType()),
            "speaking_rate_indices": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_indices_pred": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_logits": NeuralType(('B', 'C'), LogitsType()),
            "align_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        },
    )
    def forward(
        self,
        text,
        text_len,
        audio_tokens,
        audio_len,
    ):
        # Get duration information
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_len).detach()
        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, dur_len, align_hard, align_soft, align_logits = self.aligner(
            text=text,
            text_len=text_len,
            audio_codes=audio_codes,
            audio_len=audio_len,
        )
        durs = durs.detach()
        dur_len = dur_len.detach()
        speaking_rate, speaking_rate_indices = self.get_speaking_rate(text_len=text_len, durs=durs, dur_len=dur_len)
        speaking_rate = speaking_rate.detach()
        speaking_rate_indices = speaking_rate_indices.detach()

        if self.training:
            (
                text_sample,
                dur_sample,
                text_sample_len,
                audio_token_sample,
                audio_codes_sample,
                audio_token_sample_len,
                context_codes,
                context_len,
            ) = self.sample_context_audio_batch(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_len,
                text=text,
                text_lens=text_len,
                durs=durs,
            )
        else:
            (
                text_sample,
                dur_sample,
                text_sample_len,
                audio_token_sample,
                audio_codes_sample,
                audio_token_sample_len,
                context_codes,
                context_len,
            ) = self.sample_context_audio_start(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_len,
                text=text,
                text_lens=text_len,
                durs=durs,
                random_sample=False,
            )

        dur_indices = self.duration_to_index(durs=dur_sample, lengths=text_sample_len)

        context_emb = self.context_encoder(
            audio_codes=context_codes,
            audio_len=context_len,
        )
        speaking_rate_indices_pred, speaking_rate_logits = self.speaking_rate_predictor(context_emb=context_emb)

        # [batch_size, text_len, hidden_dim]
        text_enc, dur_len, _ = self.text_encoder(text=text_sample, text_len=text_sample_len, context_emb=context_emb)

        dur_indices_pred, dur_logits, dur_indices_pred_pre, dur_logits_pre = self.duration_decoder(
            inputs=text_enc,
            dur_len=dur_len,
            dur_indices=dur_indices,
            speaking_rate=speaking_rate,
        )

        text_enc_repeated, _ = regulate_len(dur_sample, text_enc, pace=1.0)
        audio_codes_sample = rearrange(audio_codes_sample, 'B C T -> B T C')
        semantic_tokens_pred_pre, semantic_logits_pre, semantic_tokens_pred, semantic_logits, acoustic_tokens_pred, acoustic_logits = self.decoder(
            hidden_state=text_enc_repeated,
            audio_len=audio_token_sample_len,
            audio_codes=audio_codes_sample,
        )

        return (
            audio_token_sample,
            audio_token_sample_len,
            semantic_tokens_pred,
            semantic_logits,
            semantic_tokens_pred_pre,
            semantic_logits_pre,
            acoustic_tokens_pred,
            acoustic_logits,
            text_sample_len,
            dur_indices,
            dur_indices_pred,
            dur_logits,
            dur_indices_pred_pre,
            dur_logits_pre,
            speaking_rate_indices,
            speaking_rate_indices_pred,
            speaking_rate_logits,
            align_hard,
            align_soft,
            align_logits,
        )

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_len = batch_dict.get("text_lens")

        audio_tokens = batch_dict.get("audio_tokens")
        audio_len = batch_dict.get("audio_token_lens")

        (
            audio_token_sample,
            audio_token_sample_len,
            _,
            semantic_logits,
            _,
            semantic_logits_pre,
            _,
            acoustic_logits,
            text_sample_len,
            dur_indices,
            _,
            dur_logits,
            _,
            dur_logits_pre,
            speaking_rate_indices,
            _,
            speaking_rate_logits,
            align_hard,
            align_soft,
            align_logits,
        ) = self(
            text=text,
            text_len=text_len,
            audio_tokens=audio_tokens,
            audio_len=audio_len,
        )

        semantic_tokens = audio_token_sample[:, : self.semantic_codebook_num, :]
        acoustic_tokens = audio_token_sample[:, self.semantic_codebook_num: , :]

        audio_mask = get_mask_from_lengths(audio_token_sample_len)

        semantic_token_loss = self.semantic_token_loss_fn(
            logits=semantic_logits, target_tokens=semantic_tokens, mask=audio_mask
        )
        train_semantic_token_loss = self.audio_token_loss_scale * semantic_token_loss

        semantic_token_loss_pre = self.semantic_token_loss_fn(
            logits=semantic_logits_pre, target_tokens=semantic_tokens, mask=audio_mask
        )
        train_semantic_token_loss_pre = self.audio_token_loss_scale * semantic_token_loss_pre

        acoustic_token_loss = self.acoustic_token_loss_fn(
            logits=acoustic_logits, target_tokens=acoustic_tokens, mask=audio_mask
        )
        train_acoustic_token_loss = self.audio_token_loss_scale * acoustic_token_loss

        text_mask = get_mask_from_lengths(text_sample_len)

        duration_loss = self.duration_loss_fn(logits=dur_logits, target_index=dur_indices.detach(), mask=text_mask)
        train_dur_loss = self.duration_loss_scale * duration_loss

        duration_loss_pre = self.duration_loss_fn(logits=dur_logits_pre, target_index=dur_indices.detach(), mask=text_mask)
        train_dur_loss_pre = self.duration_loss_scale * duration_loss_pre

        speaking_rate_loss = self.speaking_rate_loss_fn(
            logits=speaking_rate_logits, target_index=speaking_rate_indices.detach()
        )
        train_speaking_rate_loss = self.speaking_rate_loss_scale * speaking_rate_loss

        ctc_loss = self.forward_sum_loss_fn(attn_logprob=align_logits, in_lens=text_len, out_lens=audio_len)
        train_ctc_loss = self.aligner_ctc_loss_scale * ctc_loss

        if self.current_epoch < self.bin_loss_start_epoch:
            bin_loss_weight = 0.0
        elif self.current_epoch >= self.bin_loss_warmup_epochs:
            bin_loss_weight = 1.0
        else:
            bin_loss_weight = (self.current_epoch - self.bin_loss_start_epoch) / (
                self.bin_loss_warmup_epochs - self.bin_loss_start_epoch
            )

        bin_loss = self.bin_loss_fn(hard_attention=align_hard, soft_attention=align_soft)
        train_bin_loss = bin_loss_weight * self.aligner_bin_loss_scale * bin_loss

        loss = (
            train_semantic_token_loss
            + train_semantic_token_loss_pre
            + train_acoustic_token_loss
            + train_dur_loss
            + train_dur_loss_pre
            + train_speaking_rate_loss
            + train_ctc_loss
            + train_bin_loss
        )

        if not self.freeze_aligner:
            loss += train_ctc_loss + train_bin_loss

        metrics = {
            "t_semantic_token_loss": semantic_token_loss,
            "t_semantic_token_loss_pre": semantic_token_loss_pre,
            "t_acoustic_token_loss": acoustic_token_loss,
            "t_duration_loss": duration_loss,
            "t_duration_loss_pre": duration_loss_pre,
            "t_speaking_rate_loss": speaking_rate_loss,
            "t_ctc_loss": ctc_loss,
            "t_bin_loss": bin_loss,
        }

        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", semantic_token_loss, prog_bar=True, logger=False, sync_dist=True)

        return loss

    def validation_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_len = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_len = batch_dict.get("audio_token_lens")

        (
            audio_token_sample,
            audio_token_sample_len,
            semantic_tokens_pred,
            semantic_logits,
            semantic_tokens_pred_pre,
            semantic_logits_pre,
            acoustic_tokens_pred,
            acoustic_logits,
            text_sample_len,
            dur_indices,
            dur_indices_pred,
            dur_logits,
            dur_indices_pred_pre,
            dur_logits_pre,
            speaking_rate_indices,
            speaking_rate_indices_pred,
            speaking_rate_logits,
            _,
            _,
            _,
        ) = self(
            text=text,
            text_len=text_len,
            audio_tokens=audio_tokens,
            audio_len=audio_len,
        )

        audio_mask = get_mask_from_lengths(audio_token_sample_len)
        semantic_tokens = audio_token_sample[:, : self.semantic_codebook_num, :]
        acoustic_tokens = audio_token_sample[:, self.semantic_codebook_num: , :]

        semantic_token_loss = self.semantic_token_loss_fn(
            logits=semantic_logits, target_tokens=semantic_tokens, mask=audio_mask
        )
        semantic_token_correct = (semantic_tokens == semantic_tokens_pred) * rearrange(
            audio_mask, 'B T -> B 1 T'
        )
        semantic_token_accuracy = (
            semantic_token_correct.sum() / audio_token_sample_len.sum() / self.semantic_codebook_num
        )

        semantic_token_loss_pre = self.semantic_token_loss_fn(
            logits=semantic_logits_pre, target_tokens=semantic_tokens, mask=audio_mask
        )
        semantic_token_correct_pre = (semantic_tokens == semantic_tokens_pred_pre) * rearrange(
            audio_mask, 'B T -> B 1 T'
        )
        semantic_token_accuracy_pre = (
            semantic_token_correct_pre.sum() / audio_token_sample_len.sum() / self.semantic_codebook_num
        )

        acoustic_token_loss = self.acoustic_token_loss_fn(
            logits=acoustic_logits, target_tokens=acoustic_tokens, mask=audio_mask
        )
        acoustic_token_correct = (acoustic_tokens == acoustic_tokens_pred) * rearrange(
            audio_mask, 'B T -> B 1 T'
        )
        acoustic_token_accuracy = (
            acoustic_token_correct.sum() / audio_token_sample_len.sum() / self.acoustic_codebook_num
        )

        text_mask = get_mask_from_lengths(text_sample_len)

        duration_loss = self.duration_loss_fn(logits=dur_logits, target_index=dur_indices, mask=text_mask)
        dur_token_correct = (dur_indices == dur_indices_pred) * text_mask
        dur_token_accuracy = dur_token_correct.sum() / text_mask.sum()

        duration_loss_pre = self.duration_loss_fn(logits=dur_logits_pre, target_index=dur_indices, mask=text_mask)
        dur_token_correct_pre = (dur_indices == dur_indices_pred_pre) * text_mask
        dur_token_accuracy_pre = dur_token_correct_pre.sum() / text_mask.sum()

        speaking_rate_loss = self.speaking_rate_loss_fn(
            logits=speaking_rate_logits, target_index=speaking_rate_indices
        )
        speaking_rate_correct = speaking_rate_indices == speaking_rate_indices_pred
        speaking_rate_accuracy = speaking_rate_correct.float().mean()

        metrics = {
            "val_loss": semantic_token_loss,
            "val_semantic_token_loss": semantic_token_loss,
            "val_semantic_token_accuracy": semantic_token_accuracy,
            "val_semantic_token_loss_pre": semantic_token_loss_pre,
            "val_semantic_token_accuracy_pre": semantic_token_accuracy_pre,
            "val_acoustic_token_loss": acoustic_token_loss,
            "val_acoustic_token_accuracy": acoustic_token_accuracy,
            "val_duration_loss": duration_loss,
            "val_dur_token_accuracy": dur_token_accuracy,
            "val_duration_loss_pre": duration_loss_pre,
            "val_dur_token_accuracy_pre": dur_token_accuracy_pre,
            "val_speaking_rate_loss": speaking_rate_loss,
            "val_speaking_rate_accuracy": speaking_rate_accuracy,
        }

        dataset_loss_map = defaultdict(list)
        dataset_names = batch_dict.get("dataset_names")
        for i, dataset_name in enumerate(dataset_names):
            semantic_token_loss_pre = self.semantic_token_loss_fn(
                logits=semantic_logits_pre[i:i+1], target_tokens=semantic_tokens[i:i+1], mask=audio_mask[i:i+1]
            )
            semantic_token_loss_pre = semantic_token_loss_pre.cpu().item()
            dataset_loss_map[dataset_name].append(semantic_token_loss_pre)

        for dataset_name, loss_list in dataset_loss_map.items():
            metric_name = f"val_semantic_token_loss_pre_{dataset_name}"
            metric_loss = np.mean(loss_list)
            metrics[metric_name] = metric_loss

        self.log_dict(metrics, on_epoch=True, sync_dist=True)


    def on_before_optimizer_step(self, optimizer):
        for name, param in self.named_parameters():
            if param.grad is None and param.requires_grad:
                print(f"No gradient found for {name}")

        if self.skip_nan_gradients:
            # Iterate over the model's parameters to check gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Check for NaNs or Infs
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        # Zero out the gradients to prevent corruption
                        optimizer.zero_grad()
                        logging.warning(f'detected inf or nan values in gradients for {name}! Setting gradients to zero.')
                        return  # Skip the optimizer step

    def _setup_train_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=True,
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

    @typecheck(
        input_types={
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "lengths": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"dur_indices": NeuralType(('B', 'T_text'), TokenIndex())},
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
            "text_len": NeuralType(tuple('B'), LengthsType()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "dur_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
            "speaking_rate_indices": NeuralType(tuple('B'), TokenIndex()),
        },
    )
    def get_speaking_rate(self, text_len, durs, dur_len):
        sr_text_len = torch.clamp_min(text_len - 2, min=1)

        dur_mask = get_mask_from_lengths(dur_len)
        max_dur_len = dur_mask.shape[1]
        indices = torch.arange(max_dur_len, device=durs.device) + 1
        indices = rearrange(indices, 'T -> 1 T')
        dur_len = rearrange(dur_len, 'B -> B 1')
        dur_mask = torch.where(
            torch.logical_or(indices == dur_len, indices == (dur_len - 1)),
            torch.zeros_like(dur_mask),
            dur_mask
        )
        sr_durs = torch.clamp(durs.float(), min=1, max=self.max_token_duration)
        sr_durs = sr_durs * dur_mask
        sr_durs = sr_durs[:, 2:]
        sr_audio_len = sr_durs.sum(dim=1)
        # Add 1 for first/last chars
        sr_audio_len = sr_audio_len + 2

        fps = (-1.0) * sr_audio_len / sr_text_len.float()
        speaking_rate, speaking_rate_indices = self.speaking_rate_quantizer(inputs=fps)

        return speaking_rate, speaking_rate_indices


    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_len": NeuralType(tuple('B'), LengthsType()),
            "context_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "context_len": NeuralType(tuple('B'), LengthsType()),
            "frames_per_iter": NeuralType((), IntType(), optional=True),
            "audio_weight": NeuralType((), FloatType(), optional=True),
            "audio_topk": NeuralType((), IntType(), optional=True),
            "audio_temperature": NeuralType((), FloatType(), optional=True),
            "acoustic_topk": NeuralType((), IntType(), optional=True),
            "acoustic_temperature": NeuralType((), FloatType(), optional=True),
            "duration_weight": NeuralType((), FloatType(), optional=True),
            "duration_topk": NeuralType((), IntType(), optional=True),
            "duration_temperature": NeuralType((), FloatType(), optional=True),
            "speaking_rate": NeuralType(tuple('B'), FloatType(), optional=True),
            "silence_pad_start": NeuralType((), IntType(), optional=True),
            "silence_pad_end": NeuralType((), IntType(), optional=True),
            "min_speaking_rate": NeuralType((), IntType(), optional=True),
            "cond_layers": NeuralType((), IntType(), optional=True),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_token'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def infer(
        self,
        text,
        text_len,
        context_tokens,
        context_len,
        frames_per_iter=1,
        audio_weight=1.0,
        audio_topk=None,
        audio_temperature=1.0,
        acoustic_topk=None,
        acoustic_temperature=1.0,
        duration_weight=1.0,
        duration_topk=None,
        duration_temperature=1.0,
        speaking_rate=None,
        silence_pad_start=None,
        silence_pad_end=None,
        cond_layers=None,
        min_speaking_rate=-0.5,
        max_speaking_rate=0.5,
        max_infer_length = 750,
    ):
        context_emb = self.get_context(audio_tokens=context_tokens, audio_len=context_len)

        if speaking_rate is None:
            speaking_rate_indices, _ = self.speaking_rate_predictor(context_emb=context_emb)
            speaking_rate = self.speaking_rate_quantizer.get_codes(indices=speaking_rate_indices)
            speaking_rate = torch.clamp(speaking_rate, min=min_speaking_rate, max=max_speaking_rate)

        # [batch_size, text_len, hidden_dim]
        text_enc, dur_len, text_durs = self.text_encoder(text=text, text_len=text_len, context_emb=context_emb)
        # [batch_size, text_len]

        dur_indices = self.duration_decoder.infer(
            inputs=text_enc,
            dur_len=dur_len,
            speaking_rate=speaking_rate,
            frames_per_iter=frames_per_iter,
            infer_weight=duration_weight,
            silence_pad_start=silence_pad_start,
            silence_pad_end=silence_pad_end,
            topk=duration_topk,
            temperature=duration_temperature,
        )
        dur_mask = get_mask_from_lengths(dur_len)
        durs = self.index_to_duration(dur_indices=dur_indices, mask=dur_mask)

        text_enc_repeated, audio_len = regulate_len(durs, text_enc, pace=1.0)

        audio_len = torch.clamp_max(audio_len, max=max_infer_length)
        text_enc_repeated = text_enc_repeated[:, :max_infer_length, :]

        # [B, C_semantic, T]
        audio_tokens = self.decoder.infer(
            inputs=text_enc_repeated,
            audio_len=audio_len,
            frames_per_iter=frames_per_iter,
            vector_quantizer=self.vector_quantizer,
            infer_weight=audio_weight,
            topk=audio_topk,
            temperature=audio_temperature,
            cond_layers=cond_layers,
            acoustic_topk=acoustic_topk,
            acoustic_temperature=acoustic_temperature,
        )

        return audio_tokens, audio_len
