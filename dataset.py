from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
from pathlib import Path
from utils import wsj0mix
from collections import namedtuple
from torchaudio.datasets import LibriMix
from functools import partial
import torch

#抄audio 的conv-tasnet 的data util工具类处理方法

Batch = namedtuple("Batch", ["mix", "src", "mask"])

def get_dataset(
        dataset_type,
        root_dir,
        num_speakers,
        sample_rate,
        task=None,
        librimix_tr_split=None
):
    if dataset_type == "wsj0mix":
        train = wsj0mix.WSJ0Mix(root_dir / "tr", num_speakers, sample_rate)
        validation = wsj0mix.WSJ0Mix(root_dir / "cv", num_speakers, sample_rate)
        evaluation = wsj0mix.WSJ0Mix(root_dir / "tt", num_speakers, sample_rate)
    elif dataset_type == "librimix":
        train = LibriMix(root_dir, librimix_tr_split, num_speakers, sample_rate, task)
        validation = LibriMix(root_dir, "dev", num_speakers, sample_rate, task)
        evaluation = LibriMix(root_dir, "test", num_speakers, sample_rate, task)
    else:
        raise ValueError(f"Unexpected dataset: {dataset_type}")
    return train, validation, evaluation

def get_dataloader(
        dateset_type:str,
        root_dir: Union[str, Path],
        num_speakers: int = 2,
        sample_rate: int = 8000,
        batch_size: int = 6,
        num_workers: int = 4,
        librimix_task: Optional[str] = None,
        librimix_tr_split: Optional[str] = None
)  -> Tuple [DataLoader, DataLoader, DataLoader]:
# ):
    """Get dataloaders for training, validation, and testing.

        Returns:
            tuple: (train_loader, valid_loader, eval_loader)
    """
    trainset, valset, testset = get_dataset(dateset_type, root_dir, num_speakers, sample_rate, librimix_task, librimix_tr_split)
    train_collate_fn = get_collate_fn(dateset_type, mode='train', sample_rate=sample_rate, duration=3)
    test_collate_fn = get_collate_fn(dateset_type, mode='test', sample_rate=sample_rate, duration=3)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, collate_fn=test_collate_fn, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=test_collate_fn, num_workers=num_workers)
    return train_loader, val_loader, test_loader

#抄audio 的lightning_train里的方法

def get_collate_fn(dataset_type, mode, sample_rate=None, duration=4):
    assert mode in ["train", "test"]
    if dataset_type in ["wsj0mix", "librimix"]:
        if mode == "train":
            if sample_rate is None:
                raise ValueError("sample_rate is not given.")
            return partial(collate_fn_wsj0mix_train, sample_rate=sample_rate, duration=duration)
        return partial(collate_fn_wsj0mix_test, sample_rate=sample_rate)
    raise ValueError(f"Unexpected dataset: {dataset_type}")

def collate_fn_wsj0mix_train(samples: List[wsj0mix.SampleType], sample_rate, duration, augment=True):
    target_num_frames = int(duration * sample_rate)

    mixes, srcs, masks = [], [], []
    for sample in samples:
        mix, src, mask = _fix_num_frames(sample, target_num_frames, sample_rate, random_start=True)

        if augment: #注：只可在训练中增强
            mix, src = augment_audio(mix,src,sample_rate)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))

def _fix_num_frames(sample: wsj0mix.SampleType, target_num_frames: int, sample_rate: int, random_start=False):
    """Ensure waveform has exact number of frames by slicing or padding"""
    mix = sample[1]  # [1, time]
    src = torch.cat(sample[2], 0)  # [num_sources, time]

    num_channels, num_frames = src.shape
    num_seconds = torch.div(num_frames, sample_rate, rounding_mode="floor")
    target_seconds = torch.div(target_num_frames, sample_rate, rounding_mode="floor")
    if num_frames >= target_num_frames:
        if random_start and num_frames > target_num_frames:
            start_frame = torch.randint(num_seconds - target_seconds + 1, [1]) * sample_rate
            mix = mix[:, start_frame:]
            src = src[:, start_frame:]
        mix = mix[:, :target_num_frames]
        src = src[:, :target_num_frames]
        mask = torch.ones_like(mix)
    else:
        num_padding = target_num_frames - num_frames
        pad = torch.zeros([1, num_padding], dtype=mix.dtype, device=mix.device)
        mix = torch.cat([mix, pad], 1)
        src = torch.cat([src, pad.expand(num_channels, -1)], 1)
        mask = torch.ones_like(mix)
        mask[..., num_frames:] = 0
    return mix, src, mask

def collate_fn_wsj0mix_test(samples: List[wsj0mix.SampleType], sample_rate):
    max_num_frames = max(s[1].shape[-1] for s in samples)

    mixes, srcs, masks = [], [], []
    for sample in samples:
        mix, src, mask = _fix_num_frames(sample, max_num_frames, sample_rate, random_start=False)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))

def augment_audio(mix, src, sample_rate, noise_factor=0.005, pitch_factor=0.2):
    # 添加随机噪声
    noise = torch.randn_like(mix) * noise_factor
    augmented_mix = mix + noise

    augmented_src = src + noise.expand_as(src)

    # 改变音调（可选，需要额外实现）
    # ...

    return augmented_mix, augmented_src