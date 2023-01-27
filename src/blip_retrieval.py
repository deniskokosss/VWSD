
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
from lavis.models import load_model_and_preprocess, BlipBase, BlipRetrieval
from lavis.processors import load_processor
import torch.nn.functional as F
from transformers import BatchEncoding
from lavis.common.optims import LinearWarmupCosineLRScheduler
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


class BLIPRetrieval(nn.Module):
    def __init__(self, base_model, head='itm'):
        super().__init__()
        self.blip_model = base_model
        self.match_head = head

    def get_embeddings(self, inputs: Dict[str, torch.Tensor]):
        image = inputs["image"]
        caption = inputs["text_input"]

        image_embeds = self.blip_model.visual_encoder.forward_features(image)

        text = self.blip_model.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=self.blip_model.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.blip_model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        image_feat = F.normalize(self.blip_model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(
            self.blip_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        return image_feat, text_feat

    def _forward_matching(self, samples):
        image = samples["image"]
        caption = samples["text_input"]
        match_head = self.match_head

        image_embeds = self.blip_model.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.blip_model.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=self.blip_model.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        if match_head == "itm":
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = self.blip_model.tokenizer.enc_token_id  # extra code
            output = self.blip_model.text_encoder(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_output = self.blip_model.itm_head(output.last_hidden_state[:, 0, :])
            return itm_output

        elif match_head == "itc":
            text_output = self.blip_model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            image_feat = F.normalize(self.blip_model.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(
                self.blip_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim = (image_feat * text_feat).sum(dim=1)
            return sim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        images_shape = inputs["images"].shape
        batch_size = images_shape[0]
        num_pics = images_shape[1]
        text_input = []
        for t in inputs["text"]:
            for _ in range(num_pics):
                text_input.append(t)

        images_input = inputs["images"].reshape(
            batch_size * num_pics, images_shape[2], images_shape[3], images_shape[4]
        )  # image: (B * NUM_PICS, C, H, W)
        # (B * X, 2)
        if self.match_head == "itm":
            batch_outputs = self._forward_matching(
                {"text_input": text_input, "image": images_input}
            ).reshape(batch_size, num_pics, 2)  # (B, NUM_PICS, 2)
            batch_probas = F.softmax(batch_outputs[:, :, 1], dim=1)
        elif self.match_head == "itc":
            batch_outputs = self._forward_matching(
                {"text_input": text_input, "image": images_input}
            ).reshape(batch_size, num_pics)
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


class RetrievalDataset(torch.utils.data.Dataset):
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

    def __len__(self) -> int:
        return len(self.df)

    def _make_tokens(self, idx: int) -> BatchEncoding:
        return self.text_processor(self.df.iloc[idx][self.text_field])

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

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx, :]
        return {
            "text_input": row.loc[self.text_field],
            "image": self._get_image_tensor(row.loc['label']),
            "image_id": idx,
        }
