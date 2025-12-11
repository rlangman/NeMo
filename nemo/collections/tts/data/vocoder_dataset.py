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

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch.utils.data

import torch
import soundfile as sf

import torchaudio
from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.utils.tarred_dataset_utils import (
    create_tarred_dataset,
    process_tarred_manifest_vocoder,
    FileFilterIterator,
    TarredMetadata,
    VALID_AUDIO_FORMATS
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    filter_dataset,
    get_weighted_sampler,
    load_audio,
    sample_audio,
    stack_tensors,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils import webdataset as wds
from nemo.utils.decorators import experimental

from torch.utils.data import IterableDataset


@dataclass
class DatasetMeta:
    manifest_path: Path
    audio_dir: Path
    sample_weight: float = 1.0
    audio_tar_filepaths: Optional[List[str]] = None


@dataclass
class DatasetSample:
    dataset_name: str
    manifest_entry: dict
    audio_dir: Path


def create_vocoder_dataset(
    dataset_type: str,
    dataset_args: Optional[Dict] = None,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    is_train: bool = False,
):
    if not dataset_args:
        dataset_args = {}

    if dataset_type == "vocoder":
        return VocoderDataset(**dataset_args)
    elif dataset_type == "tarred_vocoder":
        if not is_train:
            raise ValueError("Tarred dataset should only be used for training set.")
        return TarredVocoderDataset(global_rank=global_rank, world_size=world_size, **dataset_args)
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")


def vocoder_collate_fn(batch: List[dict], feature_processors: List[FeatureProcessor], resample_rates=None):
    dataset_name_list = []
    audio_filepath_list = []
    audio_list = []
    audio_len_list = []

    for example in batch:
        dataset_name_list.append(example["dataset_name"])
        audio_filepath_list.append(example["audio_filepath"])
        audio_list.append(example["audio"])
        audio_len_list.append(example["audio_len"])

    batch_audio_len = torch.IntTensor(audio_len_list)
    audio_max_len = int(batch_audio_len.max().item())

    batch_audio = stack_tensors(audio_list, max_lens=[audio_max_len])

    if resample_rates:
        batch_audio = torchaudio.functional.resample(
            waveform=batch_audio, orig_freq=resample_rates[0], new_freq=resample_rates[1]
        )
        batch_audio_len = torch.ceil(batch_audio_len / resample_rates[0] * resample_rates[1]).int()
        batch_audio = mask_sequence_tensor(batch_audio, batch_audio_len)

    batch_dict = {
        "dataset_names": dataset_name_list,
        "audio_filepaths": audio_filepath_list,
        "audio": batch_audio,
        "audio_lens": batch_audio_len,
    }

    for feature_processor in feature_processors:
        feature_dict = feature_processor.collate_fn(batch)
        batch_dict.update(feature_dict)

    return batch_dict


def preprocess_manifest(
    dataset_name: str,
    dataset: DatasetMeta,
    min_duration: float,
    max_duration: float,
):
    entries = read_manifest(dataset.manifest_path)
    filtered_entries, total_hours, filtered_hours = filter_dataset(
        entries=entries, min_duration=min_duration, max_duration=max_duration
    )

    logging.info(dataset_name)
    logging.info(f"Original # of files: {len(entries)}")
    logging.info(f"Filtered # of files: {len(filtered_entries)}")
    logging.info(f"Original duration: {total_hours:.2f} hours")
    logging.info(f"Filtered duration: {filtered_hours:.2f} hours")

    samples = []
    sample_weights = []
    for entry in filtered_entries:
        sample = DatasetSample(dataset_name=dataset_name, manifest_entry=entry, audio_dir=Path(dataset.audio_dir))
        samples.append(sample)
        sample_weights.append(dataset.sample_weight)

    return samples, sample_weights


@experimental
class VocoderDataset(Dataset):
    """
    Class for processing and loading Vocoder training examples.

    Args:
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        n_samples: Optional int, if provided then n_samples samples will be randomly sampled from the full
            audio file.
        weighted_sampling_steps_per_epoch: Optional int, If provided, then data will be sampled (with replacement) based on
            the sample weights provided in the dataset metadata. If None, then sample weights will be ignored.
        feature_processors: Optional, list of feature processors to run on training examples.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        trunc_duration: Optional int, if provided audio will be truncated to at most 'trunc_duration' seconds.
        volume_norm: Whether to apply volume normalization to loaded audio.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        sample_rate: int,
        resample_rate: Optional[int] = None,
        n_samples: Optional[int] = None,
        weighted_sampling_steps_per_epoch: Optional[int] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trunc_duration: Optional[float] = None,
        volume_norm: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        if resample_rate and sample_rate != resample_rate:
            self.resample_rates = [sample_rate, resample_rate]
        else:
            self.resample_rates = None


        self.n_samples = n_samples
        self.trunc_duration = trunc_duration
        self.volume_norm = volume_norm
        self.weighted_sampling_steps_per_epoch = weighted_sampling_steps_per_epoch
        self.load_precomputed_mel = False

        if feature_processors:
            logging.info(f"Found feature processors {feature_processors.keys()}")
            self.feature_processors = list(feature_processors.values())
        else:
            self.feature_processors = []

        self.data_samples = []
        self.sample_weights = []
        for dataset_name, dataset_info in dataset_meta.items():
            dataset = DatasetMeta(**dataset_info)
            samples, weights = preprocess_manifest(
                dataset_name=dataset_name,
                dataset=dataset,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            self.data_samples += samples
            self.sample_weights += weights

    def get_sampler(self, batch_size: int, world_size: int) -> Optional[torch.utils.data.Sampler]:
        if not self.weighted_sampling_steps_per_epoch:
            return None

        sampler = get_weighted_sampler(
            sample_weights=self.sample_weights,
            batch_size=batch_size,
            world_size=world_size,
            num_steps=self.weighted_sampling_steps_per_epoch,
        )
        return sampler

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        data = self.data_samples[index]

        if self.n_samples:
            audio_array, _, audio_filepath_rel = sample_audio(
                manifest_entry=data.manifest_entry,
                audio_dir=data.audio_dir,
                sample_rate=self.sample_rate,
                n_samples=self.n_samples,
                volume_norm=self.volume_norm,
            )
        else:
            audio_array, _, audio_filepath_rel = load_audio(
                manifest_entry=data.manifest_entry,
                audio_dir=data.audio_dir,
                sample_rate=self.sample_rate,
                max_duration=self.trunc_duration,
                volume_norm=self.volume_norm,
            )
        audio = torch.tensor(audio_array, dtype=torch.float32)
        audio_len = audio.shape[0]

        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filepath_rel,
            "audio": audio,
            "audio_len": audio_len,
        }

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def collate_fn(self, batch: List[dict]):
        return vocoder_collate_fn(batch, feature_processors=self.feature_processors, resample_rates=self.resample_rates)


class TarredVocoderDataset(IterableDataset):
    """
    A similar Dataset to the VocoderDataset, but loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the VocoderDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, and duration of audio.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        n_samples: Optional int, if provided then n_samples samples will be randomly sampled from the full
            audio file.
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        trunc_duration: Optional int, if provided audio will be truncated to at most 'trunc_duration' seconds.
        feature_processors: Optional, list of feature processors to run on training examples.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        sample_rate: int,
        resample_rate: Optional[int] = None,
        sample_type: str = "concat",
        sample_args: Optional[Dict] = None,
        n_samples: Optional[int] = None,
        trunc_duration: Optional[float] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        min_duration: float = 0.1,
        max_duration: Optional[float] = None,
        volume_norm: bool = False,
        shuffle_n: int = 0,
        shuffle_n_shard: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 2,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        if resample_rate and sample_rate != resample_rate:
            self.resample_rates = [sample_rate, resample_rate]
        else:
            self.resample_rates = None

        self.n_samples = n_samples
        self.volume_norm = volume_norm
        self.load_precomputed_mel = False

        if trunc_duration:
            self.trunc_samples = int(trunc_duration * self.sample_rate)
        else:
            self.trunc_samples = None

        if feature_processors:
            logging.info(f"Found feature processors {feature_processors.keys()}")
            self.feature_processors = list(feature_processors.values())
        else:
            self.feature_processors = []

        web_datasets = []
        dataset_lengths = []
        self.file_to_sample_map = {}
        for dataset_name, dataset_info in dataset_meta.items():
            dataset_meta = TarredMetadata(**dataset_info)

            dataset_entries = read_manifest(dataset_meta.manifest_path)
            sample_map, unfiltered_file_count, unfiltered_hours, filtered_hours = process_tarred_manifest_vocoder(
                dataset_name=dataset_name,
                entries=dataset_entries,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            self.file_to_sample_map.update(sample_map)

            dataset_length = len(sample_map)
            if dataset_length == 0:
                raise ValueError(f"Found empty dataset {dataset_name} after filtering.")

            logging.info(dataset_name)
            logging.info(f"Original # of files: {len(dataset_entries)}")
            logging.info(f"Filtered # of files: {dataset_length}")
            logging.info(f"Original duration: {unfiltered_hours:.2f} hours")
            logging.info(f"Filtered duration: {unfiltered_hours:.2f} hours")

            web_dataset = self._create_web_dataset(
                tar_filepath=dataset_meta.tar_filepath,
                shuffle_n=shuffle_n,
                shuffle_n_shard=shuffle_n_shard,
                shard_strategy=shard_strategy,
                global_rank=global_rank,
                world_size=world_size
            )
            if web_dataset is not None:
                web_datasets.append(web_dataset)
                dataset_lengths.append(dataset_length)

        self.dataset = create_tarred_dataset(
            datasets=web_datasets,
            dataset_lengths=dataset_lengths,
            sample_type=sample_type,
            sample_args=sample_args
        )

        if len(self.dataset) == 0:
            raise ValueError(f"Final dataset is empty.")

    def _create_web_dataset(
        self, tar_filepath: str, shuffle_n: int, shuffle_n_shard: int, shard_strategy: str, global_rank: int, world_size: int
    ):
        tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=tar_filepath,
            global_rank=global_rank,
            world_size=world_size,
            shard_strategy=shard_strategy,
        )
        logging.info(f"Expanded {tar_filepath} to {len(tar_filepaths)} files")

        if len(tar_filepaths) == 0:
            # When using scatter shard_strategy, some workers might have no shards for a dataset
            return None

        file_ids = set(self.file_to_sample_map.keys())

        dataset = wds.DataPipeline(
            wds.SimpleShardList(urls=tar_filepaths),
            wds.shuffle(bufsize=shuffle_n_shard, initial=shuffle_n_shard),
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle_n),
            wds.rename(audio=VALID_AUDIO_FORMATS, key='__key__'),
            lambda iterator: FileFilterIterator(iterator=iterator, file_ids=file_ids),
            wds.map(self._build_sample),
        )

        return dataset

    def _build_sample(self, inputs):
        file_id = inputs["key"]
        audio_bytes = inputs["audio"]
        data = self.file_to_sample_map[file_id]

        audio_array, _ = sf.read(file=io.BytesIO(audio_bytes), dtype='float32')
        audio = torch.from_numpy(audio_array)
        if self.n_samples:
            len_audio = audio.shape[0]
            if len_audio > self.n_samples:
                start = torch.randint(0, len_audio - self.n_samples, (1,))
                audio = audio[start: start + self.n_samples]
            else:
                audio = audio[: self.n_samples]

        if self.trunc_samples:
            audio = audio[: self.trunc_samples]

        audio_len = audio.shape[0]

        audio_filepath = Path(data.manifest_entry["audio_filepath"])
        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filepath,
            "audio": audio,
            "audio_len": audio_len,
        }

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def get_sampler(self, batch_size: int, world_size: int) -> Optional[torch.utils.data.Sampler]:
        return None

    def collate_fn(self, batch):
        return vocoder_collate_fn(batch, feature_processors=self.feature_processors, resample_rates=self.resample_rates)

    def __iter__(self):
        return self.dataset.__iter__()

    def __len__(self):
        return len(self.dataset)
