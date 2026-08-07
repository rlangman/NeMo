# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This script computes audio tokens and stores them for TTS training.

$ python /NeMo/scripts/dataset_processing/tts/find_context_audio.py \
    --manifest_path=train_manifest.json \
    --output_path=train_manifest_with_context.json \
    --audio_dir=/data/audio \
    --device=cuda:0 \
    --batch_size=20000 \
    --context_min_duration=3.0 \
    --context_min_ssim=0.6 \
    --context_max_ssim=0.98

"""

import argparse
from collections import defaultdict
from itertools import batched
import json
from pathlib import Path
import random
import torch
from antlr4.error.Errors import IllegalStateException
from tqdm import tqdm

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.utils.tts_dataset_utils import load_audio, stack_tensors


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS features.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--output_path", required=True, type=Path, help="Path to output manifest.",
    )
    parser.add_argument(
        "--audio_dir", required=True, type=Path, help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to run model on.",
    )
    parser.add_argument(
        "--context_min_duration", default=3.0, type=float, help="",
    )
    parser.add_argument(
        "--context_min_ssim", default=0.6, type=float, help="",
    )
    parser.add_argument(
        "--context_max_ssim", default=0.98, type=float, help="",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="",
    )
    parser.add_argument(
        "--max_speakers_per_iter", default=10000, type=int, help="",
    )
    args = parser.parse_args()
    return args


def _compute_embeddings(rows, audio_dir, sv_model):
    audio_list = []
    audio_len_list = []
    for row in rows:
        audio_array, _, _ = load_audio(
            manifest_entry=row,
            audio_dir=audio_dir,
            sample_rate=16000,
        )
        audio = torch.from_numpy(audio_array)
        audio_list.append(audio)
        audio_len_list.append(audio.shape[0])

    max_len = max(audio_len_list)
    audio = stack_tensors(audio_list, max_lens=[max_len]).to(sv_model.device)
    audio_len = torch.IntTensor(audio_len_list).to(sv_model.device)

    with torch.no_grad():
        _, speaker_embeddings = sv_model.forward(
            input_signal=audio,
            input_signal_length=audio_len,
        )
    return speaker_embeddings


def _process_batch(rows, context_rows, min_duration, min_ssim, max_ssim, output_f):
    all_embeddings = torch.stack([r[1] for r in rows])
    context_embeddings = torch.stack([r[1] for r in context_rows])
    rows = [r[0] for r in rows]
    context_rows = [r[0] for r in context_rows]

    all_embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    context_embeddings_norm = torch.nn.functional.normalize(context_embeddings, p=2, dim=1)

    # Compute N×M similarity matrix: each row is similarities for one item against all context candidates
    similarity_matrix = torch.matmul(all_embeddings_norm, context_embeddings_norm.transpose(0, 1))

    # Sort all similarities for each item to iterate through candidates
    # sorted_similarities_tensor will contain sorted similarities for each row (original item)
    # sorted_indices_tensor will contain indices in the context_pool
    sorted_similarities_tensor, sorted_indices_tensor = torch.sort(similarity_matrix, dim=1, descending=True)

    for i, row in enumerate(rows):
        # Iterate through potential candidates from context pool, sorted by similarity
        for candidate_rank in range(sorted_indices_tensor.size(1)):
            candidate_ssim = sorted_similarities_tensor[i, candidate_rank].item()
            context_i = sorted_indices_tensor[i, candidate_rank].item()

            # If SSIM is below threshold, stop searching for this item
            if candidate_ssim < min_ssim:
                break

            # Check duration if SSIM is acceptable
            context_row = context_rows[context_i]
            candidate_duration = context_row["duration"]

            if (candidate_ssim <= max_ssim) and (candidate_duration >= min_duration):
                if row["audio_filepath"] == context_row["audio_filepath"]:
                    raise IllegalStateException(f"Selected duplicate row: {row} vs {context_row}")

                # Found a suitable candidate, update record and stop searching for this item
                record_update_dict = {
                    "context_speaker_similarity": round(candidate_ssim, 3),
                    "context_audio_filepath": context_row["audio_filepath"],
                    "context_audio_duration": candidate_duration,
                    "context_audio_text": context_row["text"],
                }
                normalized_text_candidate = context_row.get("normalized_text", None)
                if normalized_text_candidate is not None:
                    record_update_dict["context_audio_normalized_text"] = normalized_text_candidate

                output_record = row.copy()
                output_record.update(record_update_dict)
                record = json.dumps(output_record, ensure_ascii=False) + "\n"
                output_f.write(record)
                break


def main():
    args = get_args()
    manifest_path = args.manifest_path
    output_path = args.output_path
    audio_dir = args.audio_dir
    device = args.device
    batch_size = args.batch_size
    min_duration = args.context_min_duration
    min_ssim = args.context_min_ssim
    max_ssim = args.context_max_ssim
    max_speakers_per_iter = args.max_speakers_per_iter

    if not manifest_path.exists():
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not audio_dir.exists():
        raise ValueError(f"Audio directory {audio_dir} does not exist.")

    randomizer = random.Random(42)

    sv_model = EncDecSpeakerLabelModel.from_pretrained('titanet_large', map_location=device)
    sv_model = sv_model.eval()

    rows = read_manifest(manifest_path)
    speaker_row_map = defaultdict(list)
    for row in rows:
        speaker = row["speaker"]
        speaker_row_map[speaker].append(row)

    speaker_meta = list(speaker_row_map.items())
    with open(output_path, "w", encoding="utf-8") as output_f:
        for speaker, rows in tqdm(speaker_meta, miniters=100, maxinterval=600):
            speaker_embeddings = []
            for batch in batched(rows, n=batch_size):
                emb = _compute_embeddings(rows=batch, audio_dir=audio_dir, sv_model=sv_model)
                speaker_embeddings += emb

            row_with_emb = list(zip(rows, speaker_embeddings))
            for batch in batched(row_with_emb, n=max_speakers_per_iter):
                if len(row_with_emb) <= max_speakers_per_iter:
                    context_rows = row_with_emb
                else:
                    context_rows = randomizer.sample(row_with_emb, max_speakers_per_iter)
                _process_batch(
                    rows=batch,
                    context_rows=context_rows,
                    min_duration=min_duration,
                    min_ssim=min_ssim,
                    max_ssim=max_ssim,
                    output_f=output_f
            )

if __name__ == "__main__":
    main()
