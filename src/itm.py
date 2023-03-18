import logging as log
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from lavis.models import BlipBase
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List, Union, Optional
import random

from src.data import ImageSet


def infinite_repeat(value):
    while True:
        yield value


def concat_iters(*iterables):
    for it in iterables:
        for value in it:
            yield value

class ItmDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_set: ImageSet,
        text_preprocessor,
        use_context_as_text: bool = True,
        num_src_pics: int = 10,

        num_ns: int = 9,
        num_any_ns: int = 0,
        replace_any_ns: bool = False,
        replace_default_ns: bool = False,
        num_hard_ns: int = 0,
        num_any_when_no_hard_ns: int = 0,
    ) -> None:
        self.df = df
        self.image_set = image_set
        self.text_preprocessor = text_preprocessor
        self.text_field = "context" if use_context_as_text else "word"
        self.num_src_pics = num_src_pics
        self.num_ns = num_ns
        self.num_any_ns = num_any_ns
        self.replace_any_ns = replace_any_ns
        self.replace_default_ns = replace_default_ns
        self.num_hard_ns = num_hard_ns
        self.num_any_when_no_hard_ns = num_any_when_no_hard_ns

        self.all_image_names: np.ndarray = np.unique(
            self.df[[f"image{i}" for i in range(self.num_src_pics)]].values.ravel("K")
        )

        self.num_default_ns = self.num_ns - self.num_any_ns - self.num_hard_ns
        log.info(f"Total pics in sample: 1 positive, {self.num_any_ns} random from all dataset, {self.num_hard_ns} hard negative samples, {self.num_default_ns} from default rows = {self.num_ns + 1} total samples")

    def __len__(self) -> int:
        return len(self.df)

    def _sample_hard_names(self, pos_img_name: str) -> Optional[List[str]]:
        known_embs = self.image_set.known_embs
        try:
            pos_index = known_embs.index(pos_img_name)
        except ValueError:
            return None
        else:
            sim_mat = self.image_set.get_sims(known_embs)
            if sim_mat is None:
                return None
            top_indices = torch.argsort(sim_mat[pos_index], descending=True)[:self.num_hard_ns]
            return [known_embs[i] for i in top_indices]

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        pos_img_name = row["label"]

        negative_row_indices = []
        for i in range(self.num_src_pics):
            name = row[f"image{i}"]
            if name != pos_img_name:
                negative_row_indices.append(i)
        negative_row_indices = np.array(negative_row_indices)

        # making hard negatives & preparing replacements if not available
        mb_hard_ns_names = self._sample_hard_names(pos_img_name)
        if mb_hard_ns_names is None:
            add_alt_ns_num = self.num_any_when_no_hard_ns
            add_default_ns_num = self.num_hard_ns - add_alt_ns_num
            hard_ns_names = []
        else:
            add_default_ns_num = 0
            add_alt_ns_num = 0
            hard_ns_names = mb_hard_ns_names
        
        # default & alt names 
        default_ns_names = [row[f"image{i}"] for i in np.random.choice(
            negative_row_indices,
            self.num_default_ns + add_default_ns_num,
            replace = self.replace_default_ns
        )]
        alt_ns_names = list(np.random.choice(
            self.all_image_names[self.all_image_names != pos_img_name],
            self.num_any_ns + add_alt_ns_num,
            replace=self.replace_any_ns,
        ))
        
        # combine, shuffle, patch with positive
        names = default_ns_names + alt_ns_names + hard_ns_names
        assert len(names) == self.num_ns
        random.shuffle(names)
        label = random.randint(0, self.num_ns)
        names.insert(label, pos_img_name)

        return {
            "text": self.text_preprocessor(row[self.text_field]),
            "images": self.image_set[names], 
            "label": label,
            "image_names": names,
        }

class DefaultDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        text_processor,
        vis_processor,
        use_context_as_text: bool = True,
        num_pics: int = 10,
        ignore_labels: bool = False,
    ) -> None:
        self.df = df
        self.images_path = images_path
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.text_field = "context" if use_context_as_text else "word"
        self.num_pics = num_pics
        self.labels_map = None if ignore_labels else self._gen_labels()

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
        res = {
            "text": self._make_tokens(idx),
            "images": self._make_image_batch(idx), 
        }
        if self.labels_map is not None:
            res["label"] = self.labels_map[idx]
        return res 

class DefaultDatasetMultiaug(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        text_processor,
        vis_processor,
        vis_aug,
        n_aug: int = 9,
        include_original: bool = True,
        use_context_as_text: bool = True,
        num_pics: int = 10,
        ignore_labels: bool = False,
    ) -> None:
        self.df = df
        self.images_path = images_path
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.vis_aug = vis_aug
        self.n_aug = n_aug
        self.include_original = include_original
        self.text_field = "context" if use_context_as_text else "word"
        self.num_pics = num_pics
        self.labels_map = None if ignore_labels else self._gen_labels()

    def _gen_labels(self) -> Dict[int, int]: # index to label
        labels = self.df["label"].values
        zips = []
        for i in range(self.num_pics):
            images = self.df[f"image{i}"].values
            zips.append(zip(np.argwhere(labels == images).reshape(-1), infinite_repeat(i)))
        return dict(concat_iters(*tuple(zips)))
    
    def __len__(self) -> int:
        return len(self.df)

    def _make_image_tensor(self, name: str, aug: bool = False) -> torch.Tensor:
        img = Image.open(self.images_path / name).convert("RGB")
        return self.vis_processor(self.vis_aug(img) if aug else img)
    
    def _make_image_batch(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        orig = [torch.stack([self._make_image_tensor(row[f"image{i}"]) for i in range(self.num_pics)]),] if self.include_original else []
        augmented = [torch.stack([self._make_image_tensor(row[f"image{i}"], aug=True) for i in range(self.num_pics)]) for _ in range(self.n_aug)]
        return torch.stack(orig + augmented)

    def _make_tokens(self, idx: int) -> BatchEncoding:
        return self.text_processor(self.df.iloc[idx][self.text_field])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, BatchEncoding, int]]:
        # makes a batch for each row!
        res = {
            "text": self._make_tokens(idx),
            "images": self._make_image_batch(idx), 
        }
        if self.labels_map is not None:
            res["label"] = self.labels_map[idx]
        return res 

class AltNSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        text_processor,
        vis_processor,
        use_context_as_text: bool = True,
        num_negatives: int = 9,
        num_pics: int = 10,
        replace: bool = False,
    ) -> None:
        self.df = df
        self.images_path = images_path
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.text_field = "context" if use_context_as_text else "word"
        if num_negatives <= 0:
            raise ValueError(f"Exepcted num_negatives to be > 0, got {num_negatives}")
        self.num_negatives = num_negatives
        self.num_pics = num_pics
        self.replace = replace
        self.all_image_names: np.ndarray = np.unique(df[[f"image{i}" for i in range(num_pics)]].values.ravel("K"))
    
    def __len__(self) -> int:
        return len(self.df)

    def _make_image_tensor(self, name: str) -> torch.Tensor:
        return self.vis_processor(Image.open(self.images_path / name).convert("RGB"))
    
    def _make_image_batch(self, names: List[str]) -> torch.Tensor:
        return torch.stack([self._make_image_tensor(name) for name in names])
        
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        positive_image_name = row["label"]
        image_names = list(np.random.choice(
            self.all_image_names[self.all_image_names != positive_image_name],
            self.num_negatives,
            replace=self.replace
        ))
        label = random.randint(0, self.num_negatives)
        image_names.insert(label, positive_image_name)
        return {
            "text": self.text_processor(row[self.text_field]),
            "images": self._make_image_batch(image_names),
            "label": label
        }

class PersistentAltNSDataset(Dataset):
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
        use_negative_probas: bool = False,
    ) -> None:
        super().__init__()
        self.blip_model = blip_model
        self.use_negative_probas = use_negative_probas
    
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
        if self.use_negative_probas:
            # TODO: optimization
            negative_probas = outputs[:, :, 0]
            positive_probas = outputs[:, :, 1]
            mixed_logits = torch.zeros(positive_probas.shape).to(positive_probas.device)
            for batch_num in range(positive_probas.shape[0]):
                for i in range(positive_probas.shape[1]):
                    mixed_logits[batch_num][i] = positive_probas[batch_num][i] + torch.sum(negative_probas[batch_num][:i]) + torch.sum(negative_probas[batch_num][i+1:])
            return torch.softmax(mixed_logits, dim=1)
        return torch.softmax(outputs[:, :, 1], dim=1)

