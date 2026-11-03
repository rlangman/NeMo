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

$ python /NeMo/scripts/dataset_processing/tts/compute_audio_tokens.py \
    --manifest_path=train_manifest.json \
    --audio_dir=/data/audio \
    --feature_dir=/data/features \
    --feature_name="audio_tokens" \
    --model_path=/models/SpeechCodec.nemo \
    --volume_norm \
    --device=cuda:0 \
    --batch_size=16
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm

from nemo.collections.tts.parts.preprocessing.features import save_numpy_feature, _features_exists

from transformers import AutoProcessor, AutoModelForSeq2SeqLM


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS features.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--output_manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--model_name", default="google/t5gemma-2-270m-270m", type=str, help="Path to feature directory where tokens will be stored.",
    )
    parser.add_argument(
        "--feature_dir", required=True, type=Path, help="Path to feature directory where tokens will be stored.",
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        default="llm_emb",
        help="Name (directory) to store tokens under.",
    )
    parser.add_argument(
        "--device", default="gpu:0", type=str, help="Device to run model on.",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size to user during inference.",
    )
    parser.add_argument(
        "--max_entries", default=0, type=int, help="Batch size to user during inference.",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, help="Whether to overwrite existing feature files.",
    )
    args = parser.parse_args()
    return args


def _process_batch(batch_list, model, tokenizer, feature_dir, feature_name):
    text_list = [x[0] for x in batch_list]
    inputs = tokenizer(
        text=text_list,
        padding=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    token_ids = inputs["input_ids"]
    offsets = inputs["offset_mapping"]
    special_mask = inputs["special_tokens_mask"]
    attention_mask = inputs["attention_mask"]

    tokens_list = []
    for i in range(len(text_list)):
        text = text_list[i]
        tokens = []
        for token_id, (start, end), special, keep in zip(
            token_ids[i].tolist(),
            offsets[i].tolist(),
            special_mask[i].tolist(),
            attention_mask[i].tolist(),
        ):
            if not keep:
                continue
            if special:
                graphemes = "<s>"
            else:
                graphemes = text[start:end]
            tokens.append((token_id, graphemes, start, end))
        tokens_list.append(tokens)

    with torch.no_grad():
        inputs = inputs.to(model.device)
        output = model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state
        hidden_states = hidden_states.cpu().half().numpy()

    for i in range(len(text_list)):
        audio_filepath = batch_list[i][1]
        attn_mask = attention_mask[i].bool()
        hidden_state = hidden_states[i, attn_mask]

        if hidden_state.shape[0] != len(tokens_list[i]):
            print(f"Found bad alignment: {text_list[i]}, {hidden_state.shape}, {len(tokens_list[i])}, {tokens_list[i]}")

        if sum([x[3] - x[2] for x in tokens_list[i]]) != len(text_list[i]):
            print(f"Found incompatible text: {text_list[i]}, {tokens_list[i]}")

        feature_filepath = feature_dir / feature_name / audio_filepath.with_suffix(".npy")
        feature_filepath.parent.mkdir(exist_ok=True, parents=True)
        np.save(file=str(feature_filepath), arr=hidden_state)

    return tokens_list


def get_entries(manifest_path):
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as input_f:
        for line in input_f:
            entry = json.loads(line)
            if "normalized_text" in entry:
                text = entry["normalized_text"]
            else:
                text = entry["text"]
            text = text.replace("[", "")
            text = text.replace("]", "")
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            audio_filepath = entry["audio_filepath"]
            entries.append((text, Path(audio_filepath)))

    return entries


def main():
    args = get_args()
    manifest_path = args.manifest_path
    output_manifest_path = args.output_manifest_path
    model_name = args.model_name
    feature_dir = args.feature_dir
    feature_name = args.feature_name
    device = args.device
    batch_size = args.batch_size
    max_entries = args.max_entries
    overwrite = args.overwrite

    if not manifest_path.exists():
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not overwrite and output_manifest_path.exists():
        raise ValueError(f"Output manifest {output_manifest_path} exists")

    tokenizer = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device)
    model = model.get_encoder()
    model.eval()

    print(f"Reading manifest file {manifest_path}")
    entries = get_entries(manifest_path)

    if max_entries:
        entries = entries[:max_entries]

    num_entries = len(entries)
    print(f"Found {num_entries} utterances.")

    output_list = []
    batch_list = []
    for entry in tqdm(entries):
        batch_list.append(entry)
        if len(batch_list) == batch_size:
            outputs = _process_batch(
                batch_list=batch_list,
                model=model,
                tokenizer=tokenizer,
                feature_dir=feature_dir,
                feature_name=feature_name,
            )
            output_list += outputs
            batch_list = []

    if batch_list:
        outputs = _process_batch(
            batch_list=batch_list,
            model=model,
            tokenizer=tokenizer,
            feature_dir=feature_dir,
            feature_name=feature_name,
        )
        output_list += outputs

    print(f"Reading manifest file {output_manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as input_f:
        with open(output_manifest_path, "w", encoding="utf-8") as output_f:
            for i, line in enumerate(input_f):
                if max_entries and i == max_entries:
                    break
                entry = json.loads(line)
                entry["text"] = entries[i][0]
                entry["tokens"] = output_list[i]
                entry.pop("normalized_text", None)
                output_entry = f"{json.dumps(entry, ensure_ascii=False)}\n"
                output_f.write(output_entry)


if __name__ == "__main__":
    main()
