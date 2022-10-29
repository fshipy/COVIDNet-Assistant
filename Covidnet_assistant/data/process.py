import argparse
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, fields
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt
from audiomentations import (
    AddGaussianNoise,
    ClippingDistortion,
    Compose,
    PitchShift,
    PolarityInversion,
    Shift,
    TimeStretch,
    Trim,
)
from librosa import time_to_samples
from librosa.util import fix_length
from tqdm import tqdm

BAD_FILES = [
    "098d66e5-bda6-4e99-b787-ab890046c44b.mp3",
    "a9ecaf03-40a5-4b43-aaf3-f076f84a69aa.mp3",
]


@dataclass
class DataConfig:
    train_size: float
    val_size: float
    verified_only: int
    use_augmentation: bool
    use_segmentation: bool
    use_normalization: bool
    use_trim: bool
    trim_ref: int
    sample_rate: int
    augments_per_example: int
    augment_strength: int
    n_mfcc: int
    n_mels: int
    n_fft: int
    length_ms: int
    mean_size: int

    def __post_init__(self):
        assert self.n_mfcc <= self.n_mels


def augment(audio_with_meta, config):
    augments = Compose(
        [
            AddGaussianNoise(
                min_amplitude=0.001 * config.augment_strength,
                max_amplitude=0.01 * config.augment_strength,
                p=0.5,
            ),
            TimeStretch(
                min_rate=max(1 - 0.01 * config.augment_strength, 0.00001),
                max_rate=min(1 + 0.01 * config.augment_strength, 10),
                p=0.75,
            ),
            PitchShift(
                min_semitones=max(-0.2 * config.augment_strength, -12),
                max_semitones=min(0.2 * config.augment_strength, 12),
                p=0.5,
            ),
            Shift(
                min_fraction=max(-0.02 * config.augment_strength, -1),
                max_fraction=min(0.02 * config.augment_strength, 1),
                p=0.75,
            ),
            Trim(
                top_db=2 * config.augment_strength,
                p=0.75,
            ),
        ]
    )
    augmentations = {}
    logging.info("Augmenting audio")
    for fname, data in tqdm(audio_with_meta.items()):
        for i in range(config.augments_per_example):
            audio = augments(data["audio"], config.sample_rate)
            name = f"{fname}+{i}"
            augmentations[name] = {**data, "audio": audio}
    return augmentations


def audio_normalization(audio):
    return librosa.util.normalize(audio)


def feature_extraction(inputs, save_plot_path=None):
    key, raw_audio, config = inputs
    if config.use_normalization:
        raw_audio = audio_normalization(raw_audio)
    if config.use_trim:
        raw_audio, _ = librosa.effects.trim(raw_audio, ref=config.trim_ref)
    if config.use_segmentation:
        intervals = librosa.effects.split(raw_audio)
        splits = []
        for interval in intervals:
            splits = np.concatenate((splits, raw_audio[interval[0] : interval[1]]))
        raw_audio = splits
    if len(raw_audio) == 0:
        raise ValueError
    length = time_to_samples(config.length_ms / 1000, config.sample_rate)
    audio = np.asarray(fix_length(raw_audio, length), dtype=np.float32)
    if save_plot_path:
        plt.plot(raw_audio)
        plt.savefig(save_plot_path)
        plt.close()
    # extract mfcc
    audio = librosa.feature.mfcc(
        audio,
        config.sample_rate,
        dct_type=2,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=512,
        win_length=1024,
        power=1,
        lifter=0,
        htk=False,
    )

    # sample along time dimension
    audio = audio[:, : (audio.shape[1] // config.mean_size) * config.mean_size]
    return key, audio.reshape(config.n_mfcc, -1, config.mean_size).mean(-1)


def preprocess(audio_with_meta, config, parallelism, is_train):

    # augmentation
    if config.use_augmentation and is_train:
        # augment the positive ones
        augmentations = augment(audio_with_meta, config)  # positives
        audio_with_meta = {**audio_with_meta, **augmentations}
    items = [
        feature_extraction((key, data["audio"], config))
        for key, data in tqdm(audio_with_meta.items())
    ]
    for key, processed_audio in items:
        audio_with_meta[key]["audio"] = processed_audio

    return audio_with_meta


def save_data(raw_audio, metas, split, output, config, parallelism, is_train=False):
    split = set(split)
    audio_with_meta = {}
    for meta in metas:
        meta["audio"] = raw_audio[meta["filename"]]
    audio_with_meta = {
        meta["filename"]: meta for meta in metas if meta["filename"] in split
    }
    processed_audio = preprocess(audio_with_meta, config, parallelism, is_train)
    with output.open("wb") as cache:
        pickle.dump(processed_audio, cache)


def read_raw_audio(inputs):
    file_path, sample_rate = inputs
    return file_path.name, librosa.load(file_path.resolve(), sr=sample_rate)[0]


def create_datasets(
    split_file,
    cache_dir,
    data_dir,
    config,
    output_train,
    output_val,
    output_test,
    parallelism,
    seed=42,
):

    with (data_dir / "metadata.json").open("r") as meta_file:
        meta = json.load(meta_file)
        meta = [
            audio_meta for audio_meta in meta if audio_meta["filename"] not in BAD_FILES
        ]
    if split_file.is_file():
        with split_file.open("r") as split:
            splitdata = json.load(split)
            train, val, test = splitdata["train"], splitdata["val"], splitdata["test"]
    else:
        # create split
        indices = list(range(len(meta)))
        if (
            config.verified_only
        ):  # 1022 indices, include "verified = false & covid19 = false" -> 438 samples
            indices = [
                idx
                for idx, audiometa in enumerate(meta)
                if (not audiometa["covid19"]) or audiometa["verified"]
            ]
        np.random.default_rng(seed).shuffle(indices)
        mark1 = int(len(indices) * config.train_size)
        mark2 = int(len(indices) * (config.train_size + config.val_size))
        train, val, test = indices[:mark1], indices[mark1:mark2], indices[mark2:]
        train, val, test = (
            sorted([meta[idx]["filename"] for idx in train]),
            sorted([meta[idx]["filename"] for idx in val]),
            sorted([meta[idx]["filename"] for idx in test]),
        )
        with split_file.open("w") as split:
            json.dump({"train": train, "val": val, "test": test}, split)

    # cache raw audio files
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_cache = f"{config.sample_rate}.raw.cache"
    if (cache_dir / raw_audio_cache).is_file():
        with (cache_dir / raw_audio_cache).open("rb") as raw:
            raw_audio = pickle.load(raw)
    else:
        logging.info("Building Raw Audio Cache")
        files = list((data_dir / "raw").glob("*"))
        with Pool(parallelism) as pool:
            raw_audio = tqdm(
                pool.imap(
                    read_raw_audio,
                    ((file_path, config.sample_rate) for file_path in files),
                ),
                total=len(files),
            )
            raw_audio = dict(list(raw_audio))
        with (cache_dir / raw_audio_cache).open("wb") as raw:
            pickle.dump(raw_audio, raw)

    logging.info("Creating train set")
    save_data(raw_audio, meta, train, output_train, config, parallelism, is_train=True)

    logging.info("Creating validation set")
    save_data(raw_audio, meta, val, output_val, config, parallelism)

    logging.info("Creating test set")
    save_data(raw_audio, meta, test, output_test, config, parallelism)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Covid-Cough datasets.")
    parser.add_argument(
        "--split_file",
        type=Path,
        default=(Path(__file__).parent / ".split.txt").resolve(),
        help="Split file to load or if it does not exits, generate this split file.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=(Path(__file__).parent / ".cache_dir").resolve(),
        help="Directory to store sampled audio cache",
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Input audio data directory."
    )
    parser.add_argument(
        "--output_train", type=Path, required=True, help="Output train data path."
    )
    parser.add_argument(
        "--output_val", type=Path, required=True, help="Output validation data path."
    )
    parser.add_argument(
        "--output_test", type=Path, required=True, help="Output test data path."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use when splitting the dataset."
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of processes to use when processing audio.",
    )

    parser.add_argument(
        "--train_size", type=float, default=0.6, help="Size of the train split. (0-1)"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Size of the validation set. (0-1)"
    )
    parser.add_argument(
        "--verified_only",
        action="store_true",
        default=False,
        help="Use verified only set.",
    )
    parser.add_argument("--n_mels", type=int, default=32, help="Number of mels.")
    parser.add_argument(
        "--trim_ref", type=int, default=20, help="ref when trimming audios."
    )
    parser.add_argument("--n_mfcc", type=int, default=32, help="Number of mfccs.")
    parser.add_argument("--n_fft", type=int, default=2048, help="Number of ffts.")
    parser.add_argument(
        "--length_ms", type=int, default=7000, help="Maximum length of the audio."
    )
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sample rate.")
    parser.add_argument("--mean_size", type=int, default=2, help="Mean size to use.")
    parser.add_argument(
        "--augments_per_example",
        type=int,
        default=1,
        help="Augmented audio per input audio.",
    )
    parser.add_argument(
        "--augment_strength",
        type=int,
        default=10,
        help="Augmentation strength.",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=False,
        help="Use augmentation or not.",
    )
    parser.add_argument(
        "--use_segmentation",
        action="store_true",
        default=False,
        help="Use segmentation or not. This will remove silent intervals in the audio pieces.",
    )
    parser.add_argument(
        "--use_trim",
        action="store_true",
        default=False,
        help="Use trim or not.",
    )
    parser.add_argument(
        "--use_normalization",
        action="store_true",
        default=False,
        help="Use normalization or not.",
    )
    args = parser.parse_args()
    # assert args.augment_strength <= 50, "augment_strength should be no larger than 50"
    data_config = DataConfig(
        **{
            k.name: getattr(args, k.name)
            for k in fields(DataConfig)
            if k.name in vars(args)
        }
    )

    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    logging.info("cache dir: " + str(args.cache_dir))
    create_datasets(
        args.split_file,
        args.cache_dir,
        args.data_dir,
        data_config,
        args.output_train,
        args.output_val,
        args.output_test,
        args.parallelism,
        seed=args.seed,
    )
