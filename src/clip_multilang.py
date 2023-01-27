
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
from transformers import BatchEncoding, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from sentence_transformers import SentenceTransformer

from src.data import CustomSplitLoader
from src.utils import evaluate, mrr


def infinite_repeat(value):
    while True:
        yield value


def concat_iters(*iterables):
    for it in iterables:
        for value in it:
            yield value


def to_device(object, device):
    if not isinstance(object, dict):
        raise NotImplementedError("Implement other types than dict if needed!")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in object.items()}

# class CustomCLIPM(pt_multilingual_clip.MultilingualCLIP):
#     def forward(self, txt, tokenizer):
#         txt_tok = tokenizer(txt, padding=True, return_tensors='pt')
#         txt_tok = txt_tok.to(torch.device('cuda'))
#         embs = self.transformer(**txt_tok)[0]
#         att = txt_tok['attention_mask']
#         embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
#         return self.LinearTransformation(embs)


class Classifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_negative: int = 9,
    ) -> None:
        super().__init__()
        self.text_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32-multilingual-v1'
        )
        self.image_model = SentenceTransformer(
            'clip-ViT-B-32'
        )
        self.num_pics = num_negative + 1

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: move all this batch-dependent stuff to collate_fn?
        # TODO: optimize!
        # text: str
        # image:
        # image: (B, NUM_PICS, C, H, W)
        batch_size = len(inputs['images']) // self.num_pics
        images_input = []
        for batch in inputs['images']:
            images_input += batch
        text_input = inputs['text']

        # TODO: 10 -> X
        # images_input = inputs["images"].reshape(
        #     batch_size * self.num_pics, images_shape[2], images_shape[3], images_shape[4]
        # )  # image: (B * NUM_PICS, C, H, W)
        # (B * X, 2)
        # text_input = self.text_model.tokenize(text_input)
        text_input = to_device(self.text_model.tokenize(text_input), self.text_model.device)
        text_embedding = self.text_model.forward(text_input)['sentence_embedding']

        images_input = to_device(dict(self.image_model.tokenize(images_input)), self.text_model.device)
        image_embedding = self.image_model.forward(images_input)['sentence_embedding']

        sims = 6 * torch.bmm(
            image_embedding.view(batch_size, self.num_pics, -1),
            text_embedding.unsqueeze(-1)
        ).squeeze(-1)
        return F.softmax(sims, dim=-1)



class ClipDataset(torch.utils.data.Dataset):
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
        return self.vis_processor([Image.open(path).convert("RGB")])

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
        return [self._get_image_tensor(row[f"image{i}"]) for i in range(self.num_pics)]

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