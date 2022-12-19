from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from lavis.models import BlipBase
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List, Union
import random


def infinite_repeat(value):
    while True:
        yield value


def concat_iters(*iterables):
    for it in iterables:
        for value in it:
            yield value


class DefaultDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        text_processor,
        vis_processor,
        use_context_as_text: bool = True,
        num_pics: int = 10,
    ) -> None:
        self.df = df
        self.images_path = images_path
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.text_field = "context" if use_context_as_text else "word"
        self.num_pics = num_pics
        self.labels_map = self._gen_labels()

    def _gen_labels(self) -> Dict[int, int]: # index to label
        labels = self.df["label"].values
        zips = []
        for i in range(self.num_pics):
            images = self.df[f"image{i}"].values
            zips.append(zip(np.argwhere(labels == images).reshape(-1), infinite_repeat(i)))
        return dict(concat_iters(*tuple(zips)))
    
    def __len__(self) -> int:
        return len(self.df)

    def _make_image_tensor(self, name: str) -> torch.Tensor:
        return self.vis_processor(Image.open(self.images_path / name).convert("RGB"))
    
    def _make_image_batch(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        return torch.stack([self._make_image_tensor(row[f"image{i}"]) for i in range(self.num_pics)])

    def _make_tokens(self, idx: int) -> BatchEncoding:
        return self.text_processor(self.df.iloc[idx][self.text_field])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, BatchEncoding, int]]:
        # makes a batch for each row!
        return {
            "text": self._make_tokens(idx),
            "images": self._make_image_batch(idx),
            "label": self.labels_map[idx],
        }

# TODO: should make new negative sampling each time without persistence?
class AltNSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        text_processor,
        vis_processor,
        use_context_as_text: bool = True,
        num_negatives: int = 10,
        num_pics: int = 10,
        replace: bool = False,
    ) -> None:
        self.text_processor = text_processor
        self.images_path = images_path
        self.vis_processor = vis_processor
        text_field = "context" if use_context_as_text else "word"
        self.samples = []
        if num_negatives <= 0:
            raise ValueError(f"Exepcted num_negatives to be > 0, got {num_negatives}")
        all_image_names: np.ndarray = np.unique(df[[f"image{i}" for i in range(num_pics)]].values.ravel("K"))
        for _, row in df.iterrows():
            positive_image_name = row["label"]
            image_names = list(np.random.choice(all_image_names[all_image_names != positive_image_name], num_negatives, replace=replace))
            label = random.randint(0, num_negatives)
            image_names.insert(label, positive_image_name)
            self.samples.append({
                "text": row[text_field],
                "image_names": image_names,
                "label": label,
            })
    
    def __len__(self) -> int:
        return len(self.samples)

    def _make_image_tensor(self, name: str) -> torch.Tensor:
        return self.vis_processor(Image.open(self.images_path / name).convert("RGB"))
    
    def _make_image_batch(self, names: List[str]) -> torch.Tensor:
        return torch.stack([self._make_image_tensor(name) for name in names])
        
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        sample = self.samples[idx]
        return {
            "text": self.text_processor(sample["text"]),
            "images": self._make_image_batch(sample["image_names"]),
            "label": sample["label"]
        }

def to_device(obj, device):
    if not isinstance(obj, dict):
        raise NotImplementedError("Implemented only for dict structures")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}


class ITMClassifier(nn.Module):
    def __init__(
        self,
        blip_model: BlipBase,
    ) -> None:
        super().__init__()
        self.blip_model = blip_model
    
    def forward(self, inputs: Dict[str, Union[str, torch.Tensor]]) -> torch.Tensor:
        """
        Note, that forward can be done for variable number of images
        """
        images = inputs["images"]
        text = inputs["text"]
        bs = images.shape[0]
        n = images.shape[1]
        c = images.shape[2]
        h = images.shape[3]
        w = images.shape[4]
        len_flat = bs * n
        texts = []
        for t in text:
            for _ in range(n):
                texts.append(t)
        outputs = self.blip_model({"text_input": texts, "image": images.reshape(len_flat, c, h, w)}, match_head = "itm").reshape(bs, n, 2)
        return torch.softmax(outputs[:, :, 1], dim=1)

