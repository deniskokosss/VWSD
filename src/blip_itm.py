
from pathlib import Path
import logging
import json
from typing import *
import time
import sys

sys.path.append('..')
sys.path.append('.')

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
import torch.nn as nn
from lavis.models import load_model_and_preprocess, BlipBase
from lavis.processors import load_processor
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score

from src.data import CustomSplitLoader
from src.utils import evaluate, mrr


def infinite_repeat(value):
    while True:
        yield value


def concat_iters(*iterables):
    for it in iterables:
        for value in it:
            yield value

class Classifier(nn.Module):
    def __init__(
        self,
        blip_model: BlipBase,
        match_head: str = "itm",
        head_sum_bias_enabled: bool = True,
        num_negative: int = 9,
    ) -> None:
        super().__init__()
        self.blip_model = blip_model
        self.match_head = match_head
        self.num_pics = num_negative + 1
        if self.match_head == "mean":
            self.head_combiner = nn.Linear(2, 1, bias=head_sum_bias_enabled)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: move all this batch-dependent stuff to collate_fn?
        # TODO: optimize!
        # text: str
        # image:
        images_shape = inputs["images"].shape # image: (B, NUM_PICS, C, H, W)
        batch_size = images_shape[0]
        text_input = []
        for t in inputs["text"]:
            for _ in range(self.num_pics):
                text_input.append(t)
        # TODO: 10 -> X
        images_input = inputs["images"].reshape(
            batch_size * self.num_pics, images_shape[2], images_shape[3], images_shape[4]
        )  # image: (B * NUM_PICS, C, H, W)
        # (B * X, 2)
        if self.match_head == "itm":
            batch_outputs = self.blip_model(
                {"text_input": text_input, "image": images_input},
                match_head=self.match_head
            ).reshape(batch_size, self.num_pics, 2)  # (B, NUM_PICS, 2)
            batch_probas = F.softmax(batch_outputs[:, :, 1], dim=1)
        elif self.match_head == "itc":
            batch_outputs = self.blip_model(
                {"text_input": text_input, "image": images_input},
                match_head=self.match_head
            ).reshape(batch_size, self.num_pics) # (B * NUM_PICS) -> (B, NUM_PICS)
            # hugginface VisionTextEncoder see cos * N before softmax
            # TODO: N as hyperparam const
            # TODO: or learnable param
            batch_probas = F.softmax(batch_outputs, dim=1) # softmax(cosine similarity) => =(
        elif self.match_head == "mean":
            raise NotImplementedError("Implement me!")
            # Warning: not tested
            # TODO: replace with mean(p_itm, p_itc)
        else:
            raise ValueError(
                f"Unexpected value for match_head parameter"
                "{self.match_head}\". Allowed values: \"itm\", \"itc\" or \"mean\"."
            )
        return batch_probas


class ItmDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            images_path: Path,
            text_processor,
            vis_processor,
            folder_struct: Optional[dict[str, list[str]]] = None,
            use_context_as_text: bool = True,
            enable_cache: bool = False,
            num_negative: int = 9,
    ) -> None:
        self.df = df
        self.images_path = images_path
        self.folder_struct = folder_struct
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.tokens_cache = dict()
        self.image_tensor_cache = dict()
        self.enable_cache = enable_cache
        self.num_pics = num_negative + 1
        self.text_field = "context" if use_context_as_text else "word"
        self.labels_map = self._gen_labels()

    def _gen_labels(self) -> Dict[int, int]:  # index to label
        labels = self.df["label"].values
        zips = []
        for i in range(self.num_pics):
            images = self.df[f"image{i}"].values
            zips.append(zip(np.argwhere(labels == images).reshape(-1), infinite_repeat(i)))
        return dict(concat_iters(*tuple(zips)))

    def __len__(self) -> int:
        return len(self.df)

    def _make_image_tensor(self, name: str) -> torch.Tensor:
        path = Path(self.images_path)
        if self.folder_struct:
            for fld in self.folder_struct[name]:
                path = path / fld
        path = path / name
        return self.vis_processor(Image.open(path).convert("RGB"))

    def _get_image_tensor(self, name: str) -> Image:
        if not self.enable_cache:
            return self._make_image_tensor(name)
        if name in self.image_tensor_cache:
            return self.image_tensor_cache[name]
        t = self._make_image_tensor(name)
        self.image_tensor_cache[name] = t
        return t

    def _get_image_batch(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        return torch.stack([self._get_image_tensor(row[f"image{i}"]) for i in range(self.num_pics)])

    def _make_tokens(self, idx: int) -> BatchEncoding:
        return self.text_processor(self.df.iloc[idx][self.text_field])

    def _get_tokens(self, idx: int) -> BatchEncoding:
        if not self.enable_cache:
            return self._make_tokens(idx)
        if idx in self.tokens_cache:
            return self.tokens_cache[idx]
        t = self._make_tokens(idx)
        self.tokens_cache[idx] = t
        return t

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, BatchEncoding, int]]:
        # makes a batch for each row!
        return {
            "text": self._get_tokens(idx),
            "images": self._get_image_batch(idx),
            "label": self.labels_map[idx],
        }
