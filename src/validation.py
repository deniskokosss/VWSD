from typing import Dict, Any
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import top_k_accuracy_score
from src.utils import mrr
import numpy as np

class Validation:
    def __init__(self, common: Dict[str, Any] = {}, configs: Dict[str, Dict[str, Any]] = {}):
        self.common = common
        self.configs = configs

    def __getitem__(self, idx):
        return dict(self.common, **self.configs[idx])

    @property
    def names(self):
        return self.configs.keys()
    
    def _validate(self, model: nn.Module, dev, dl: DataLoader, f_batch: [[nn.Module, Any, torch.device, Dict[str, Any]]], env) -> Dict[str, float]:
        scores = dict()
        for batch in dl:
            for k, v in f_batch(model, batch, dev, env).items():
                if k in scores: scores[k] += v
                else: scores[k] = v
        l = len(dl)
        return {k: v / l for k, v in scores.items()}

    def __call__(self, train_step, model):
        acc = dict()
        model.eval()
        for name in self.names:
            d = self[name]
            if d.setdefault("enable", True) and d.setdefault("step_cond", lambda _: True)(train_step):
                d.setdefault("log_step", lambda s, n: None)(train_step, name)
                model = model.to(d["device"])
                do_validate = lambda: self._validate(
                    model = model,
                    dev = d["device"],
                    dl = d["dl"],
                    f_batch = d["get_batch_scores"],
                    env = d,
                )
                if d.setdefault("no_grad", True):
                    with torch.no_grad():
                        scores = do_validate()
                else:
                    scores = do_validate()
                acc[name] = scores
                for k, v in scores.items():
                    d["log_score"](train_step, name, k, v)
        model.train()
        return acc

# TODO: refactor? | remove? code below

metric2name = {
    "acc1": "Accuracy@Top1",
    "acc3": "Accuracy@Top3",
    "mrr": "Mean Reciprocal Rank",
}

def eval_batch(labels, preds, num_labels):
    labels_range = np.arange(num_labels)
    labels = labels.numpy(force=True)
    preds = preds.numpy(force=True)
    return {
        "acc1": top_k_accuracy_score(labels, preds, k=1, labels=labels_range), 
        "acc3": top_k_accuracy_score(labels, preds, k=3, labels=labels_range),
        "mrr": mrr(labels, preds),
    }

def sum_scores(scores, new_scores):
    return {k: scores[k] + new_scores[k] for k in scores}

def div_scores(scores, n):
    return {k: v / n for k, v in scores.items()}