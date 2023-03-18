import pandas as pd
from dataclasses import dataclass
from typing import *
from pathlib import Path
from logging import warn 
import numpy as np
from functools import reduce
from PIL import Image
import torch
import torch.nn as nn

class CustomSplitLoader:
    def __init__(
        self,
        split_parts: Dict[str, float],
        data_path: Path,
        labels_path: Path,
        random_state: int = 123,
        isolate_words: bool = True,
    ) -> None:
        if sum(split_parts.values()) != 1:
            raise ValueError("Split parts must sum up to 1")
        self.split_parts = split_parts
        self.data_path = data_path
        self.labels_path = labels_path
        self.random_state = random_state
        if isolate_words != True:
            warn("Mind that train / test samples would are isolated during actual competition, so tests with such a split might be less representative!")
        self.isolate_words = isolate_words

    def make_simple_splits(self, df) -> Dict[str, Set[int]]:
        index = np.array(df.index)
        l = len(index)
        np.random.default_rng(self.random_state).shuffle(index)
        relative_borders = reduce(lambda acc, v: acc + [acc[-1] + v,], self.split_parts.values(), [0,])
        index_splits = {k: index[int(relative_borders[i] * l):int(relative_borders[i + 1] * l)] for (i, k) in enumerate(self.split_parts.keys())}
        return {part: df.loc[indices] for part, indices in index_splits.items()} 

    def make_word_isolated_splits(self, df) -> Dict[str, Set[int]]:
        words = np.array(df["word"].unique())
        l = len(words)
        np.random.default_rng(self.random_state).shuffle(words)
        relative_borders = reduce(lambda acc, v: acc + [acc[-1] + v,], self.split_parts.values(), [0,])
        absolute_borders = [int(p * l) for p in relative_borders]
        words_split = {k: words[absolute_borders[i]:absolute_borders[i+1]] for (i, k) in enumerate(self.split_parts.keys())}
        return {k: df[df["word"].apply(lambda w: w in ws)] for k, ws in words_split.items()}

    def get_splits(self,) -> Dict[str, pd.DataFrame]:
        df = pd.read_csv(self.data_path, sep = "\t", header=None)
        df.columns = ["word", "context"] + [f"image{i}" for i in range(10)]
        df["label"] = pd.read_csv(self.labels_path, sep="\t", header=None)
        return self.make_word_isolated_splits(df) if self.isolate_words else self.make_simple_splits(df)

class ImageSet:
    def __init__(
        self,
        images_path: Path,
        image_processor: Callable[[Image], torch.Tensor],
        similarity_measure: Callable[[torch.Tensor], torch.Tensor] = nn.CosineSimilarity(dim=1),
        enable_cache: bool = True,
    ) -> None:
        self.images_path = images_path
        self.image_processor = image_processor
        self.enable_cache = enable_cache
        self.similarity_measure = similarity_measure
        self.tensor_cache: Dict[str, torch.Tensor] = dict() # <file name> -> <data>
        self.embedding_cache: Dict[str, torch.Tensor] = dict() # <file name> -> <embedding>
        self.similarities_cache: Dict[str, Dict[str, float]] = dict() # fn1 -> fn2 -> sim(fn1, fn2)

    def __getitem__(self, file_name: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(file_name, list):
            return torch.stack([self[n] for n in file_name])

        if file_name in self.tensor_cache:
            return self.tensor_cache[file_name]
        loaded = self.image_processor(Image.open(self.images_path / file_name).convert("RGB"))
        if self.enable_cache:
            self.tensor_cache[file_name] = loaded
        return loaded

    @property
    def known_embs(self) -> List[str]:
        return list(self.embedding_cache.keys())

    def update_emb(self, file_name: str, vec: torch.Tensor):
        self.embedding_cache[file_name] = vec

    def get_emb(self, file_name: str) -> Optional[torch.Tensor]:
        try:
            return self.embedding_cache[file_name]
        except:
            return None
    
    def get_sims(self, file_names: List[str]) -> Optional[torch.Tensor]:
        embeddings = []
        for name in file_names:
            emb = self.get_emb(name)
            if emb is None:
                return None
            embeddings.append(emb)
        embeddings = torch.stack(embeddings)
        return self.similarity_measure(embeddings)
        
