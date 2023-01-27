# %% [markdown]
# # BLIP finetuning to target task
# 
# TODO: rewrite this to reflect the latest changes
# 
# Sample is formed from a single row of dataset:
# $$\operatorname{batch} = ((E_t, E_{i_0}), (E_t, E_{i_1}), ..., (E_t, E_{i_9})); \operatorname{batch} : R^{10 \times (E_t + E_i)}$$
# ITM predicts probas for $y = 0$, $y = 1$
# $$\operatorname{ITM} : R^{10 \times (E_t + E_i)} \rightarrow R^{10 \times 2}$$
# Model is defined as:
# $$\operatorname{F} = \operatorname{softmax} \circ \operatorname{ITM} \circ \operatorname{batch}$$
# $$\operatorname{F} : R^{10 \times (E_t + E_i)} \rightarrow R^{10}$$
# So, this definition is for a single row
import os
from pathlib import Path
import logging
import json
from typing import *
import time
import sys

import open_clip

sys.path.append('../..')
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
import torch.nn as nn
from lavis.models import load_model_and_preprocess, BlipBase
from lavis.processors import load_processor
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers import BatchEncoding
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import clip


from src.data import CustomSplitLoader
from src.utils import evaluate, mrr
from src.blip_itm import ItmDataset
from src.clip_multilang import ClipDataset as ItmDataset
from src.blip_itm import Classifier as ClassifierBLIP

from src.clip_multilang import Classifier as ClassifierCLIPM


def to_device(object, device):
    if not isinstance(object, dict):
        raise NotImplementedError("Implement other types than dict if needed!")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in object.items()}


def collate_fn(batches):
    res = {
        "text": [],
        "images": [],
        "label": [],
    }
    for batch in batches:
        res['text'].append(batch['text'])
        res['images'] += batch['images']
        res['label'].append(batch['label'])
    res['label'] = torch.as_tensor(res['label'])
    return res

def do_nothing(x):
    return x

def convert_to_tensor(x):
    return torch.as_tensor(np.array(x))


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    DEVICE = torch.device(cfg.DEVICE)
    os.makedirs(cfg.SAVE_CHECKPOINT_PATH, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # some images from train might not load without the following settings or warnings would be thrown
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    pth = Path(HydraConfig.get().runtime.output_dir)
    writer = SummaryWriter(pth)

    print(f"OUTPUT_DIR={pth}")

    print(f"Running on {DEVICE}")

    df = pd.read_csv(cfg.data.DATA_PATH, sep='\t', header=None)
    df.columns = ["word", "context"] + [f"image{i}" for i in range(cfg.model.NUM_PICS)]
    df["label"] = pd.read_csv(cfg.data.LABELS_PATH, sep='\t', header=None)

    if cfg.DO_TEST:
        test_df = df.loc[pd.read_csv(cfg.data.TEST_SPLIT_PATH, sep='\t', header=None).T.values[0]]

        test2_df = pd.read_csv(cfg.data.TEST2_PATH, sep='\t', header=None)
        test2_df.columns = ["word", "context"] + [f"image{i}" for i in range(cfg.model.NUM_PICS)]
        test2_df["label"] = pd.read_csv(cfg.data.TEST2_LABELS_PATH, sep='\t', header=None)

    if cfg.DO_TRAINING:
        validation_df = df.loc[pd.read_csv(cfg.data.VALIDATION_SPLIT_PATH, sep='\t', header=None).T.values[0]]

        validation2_df = pd.read_csv(cfg.data.VALID2_PATH, sep='\t', header=None)
        validation2_df.columns = ["word", "context"] + [f"image{i}" for i in range(cfg.model.NUM_PICS)]
        validation2_df["label"] = pd.read_csv(cfg.data.VALID2_LABELS_PATH, sep='\t', header=None)

        validation5_df = pd.read_csv(cfg.data.VALID5_PATH, sep='\t', header=None)
        validation5_df.columns = ["word", "context"] + [f"image{i}" for i in range(cfg.model.NUM_PICS)]
        validation5_df["label"] = pd.read_csv(cfg.data.VALID5_LABELS_PATH, sep='\t', header=None)

        if cfg.data.TYPE == 'wikidata':
            train_df = pd.read_csv(cfg.data.TRAIN_PATH, sep='\t', header=None)
            train_df.columns = ["word", "context"] + [f"image{i}" for i in range(cfg.model.NUM_PICS)]
            train_df["label"] = pd.read_csv(cfg.data.TRAIN_LABELS_PATH, sep='\t', header=None)
        elif cfg.data.TYPE == 'default':
            train_df = df.loc[pd.read_csv(cfg.data.TRAIN_SPLIT_PATH, sep='\t', header=None).T.values[0]]
    if 'clip' not in cfg.model.BLIP_VARIANT:
        blip_model, vis_processors, text_processors = load_model_and_preprocess(
            "blip_image_text_matching", cfg.model.BLIP_VARIANT, is_eval=True
        )
    else:
        model = ClassifierCLIPM(cfg.model.BLIP_VARIANT).to(DEVICE)
        text_processors = {'eval': do_nothing}
        vis_processors = {'eval': do_nothing}
    if cfg.data.IMAGES_FOLDER_STRUCT:
        folder_struct = json.load(open(cfg.data.IMAGES_FOLDER_STRUCT, 'r'))
    else:
        folder_struct = None
    if cfg.DO_TRAINING:
        train_ds = ItmDataset(
            df=train_df,
            images_path=cfg.data.IMAGES_PATH,
            folder_struct=folder_struct,
            text_processor=text_processors["eval"],
            vis_processor=vis_processors["eval"],
        )
        val_ds = ItmDataset(
            df=validation_df,
            images_path=cfg.data.IMAGES_PATH,
            folder_struct=folder_struct,
            text_processor=text_processors["eval"],
            vis_processor=vis_processors["eval"],
        )
        val2_ds = ItmDataset(
            df=validation2_df,
            images_path=cfg.data.IMAGES_PATH,
            folder_struct=folder_struct,
            text_processor=text_processors["eval"],
            vis_processor=vis_processors["eval"],
        )
        val5_ds = ItmDataset(
            df=validation5_df,
            images_path=cfg.data.IMAGES_PATH,
            folder_struct=folder_struct,
            text_processor=text_processors["eval"],
            vis_processor=vis_processors["eval"],
        )

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.train.TRAIN_BATCH_SIZE, collate_fn=collate_fn,
                                               num_workers=cfg.NUM_WORKERS, persistent_workers=True, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.train.VALIDATION_BATCH_SIZE, collate_fn=collate_fn,
                                             num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        val2_dl = torch.utils.data.DataLoader(val2_ds, batch_size=cfg.train.VALIDATION_BATCH_SIZE, collate_fn=collate_fn,
                                              num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        val5_dl = torch.utils.data.DataLoader(val5_ds, batch_size=cfg.train.VALIDATION_BATCH_SIZE, collate_fn=collate_fn,
                                              num_workers=cfg.NUM_WORKERS, persistent_workers=True)

    if cfg.DO_TEST:
        test_ds = ItmDataset(
            df=test_df,
            images_path=cfg.data.IMAGES_PATH,
            folder_struct=folder_struct,
            text_processor=text_processors["eval"],
            vis_processor=vis_processors["eval"],
        )
        test2_ds = ItmDataset(
            df=test2_df,
            images_path=cfg.data.IMAGES_PATH,
            folder_struct=folder_struct,
            text_processor=text_processors["eval"],
            vis_processor=vis_processors["eval"],
        )
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=1, num_workers=cfg.NUM_WORKERS, persistent_workers=True, shuffle=False
        )
        test2_dl = torch.utils.data.DataLoader(
            test2_ds, batch_size=1, num_workers=cfg.NUM_WORKERS, persistent_workers=True, shuffle=False)
    if 'clip' not in cfg.model.BLIP_VARIANT:
        model = ClassifierBLIP(blip_model).to(DEVICE)

    if cfg.model.CHECKPOINT:
        p = Path(cfg.model.CHECKPOINT)
        dict_ = torch.load(p)
        model.load_state_dict(dict_['model'])

        logging.info(f"Loaded {p}")

    metric2name = {
        "acc1": "Accuracy@Top1",
        "acc3": "Accuracy@Top3",
        "mrr": "Mean Reciprocal Rank",
    }

    labels_range = np.arange(cfg.model.NUM_PICS)

    def eval_batch(labels, preds):
        labels = labels.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        return {
            "acc1": top_k_accuracy_score(labels, preds, k=1, labels=labels_range), 
            "acc3": top_k_accuracy_score(labels, preds, k=3, labels=labels_range),
            "mrr": mrr(labels, preds),
        }

    def sum_scores(scores, new_scores):
        return {k: scores[k] + new_scores[k] for k in scores}

    def div_scores(scores, n):
        return {k: v / n for k, v in scores.items()}

    train_ds[0]

    loss_fn = nn.CrossEntropyLoss()
    if cfg.DO_TRAINING:
        TRAIN_EFFECTIVE_BATCH_SIZE = cfg.train.GRAD_ACCUM_STEPS * cfg.train.TRAIN_BATCH_SIZE

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.LR, weight_decay=cfg.train.WEIGHT_DECAY)
        num_training_steps = int(cfg.train.NUM_EPOCHS * (len(train_dl) / cfg.train.GRAD_ACCUM_STEPS))
        num_warmup_steps = int(num_training_steps * cfg.train.WARMUP_STEPS_FRAC)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"{num_training_steps} training steps which include {num_warmup_steps} warmup ones")

        step_num = 0
        steps_since_last_eval = cfg.train.STEPS_BETWEEN_EVAL - TRAIN_EFFECTIVE_BATCH_SIZE - 1
        grad_accum_step_cnt = 0
        save_checkpoint_step_cnt = 0

        if cfg.model.CHECKPOINT:
            p = Path(cfg.SAVE_CHECKPOINT_PATH) / f"step-{cfg.model.CHECKPOINT}.pt"
            dict_ = torch.load(p)
            model.load_state_dict(dict_['model'])
            step_num = cfg.train.START_FROM
            logging.info(f"Loaded {p}")
            optimizer.load_state_dict(dict_['optimizer'])
            lr_scheduler.load_state_dict(dict_['scheduler'])
            del dict_

        progress_bar = tqdm(range(num_training_steps))
        for epoch_num in range(cfg.train.NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            train_scores = {"acc1": 0, "acc3": 0, "mrr": 0}
            for batch in train_dl:
                outputs = model(to_device(batch, DEVICE))
                loss = loss_fn(outputs, F.one_hot(batch["label"], 10).float().to(DEVICE))
                train_loss += loss.item()
                new_scores = eval_batch(batch["label"], outputs)
                train_scores = sum_scores(train_scores, new_scores)
                loss.backward()
                grad_accum_step_cnt += 1

                if grad_accum_step_cnt == cfg.train.GRAD_ACCUM_STEPS:
                    writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], step_num)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    writer.add_scalar("Loss/Train", train_loss / TRAIN_EFFECTIVE_BATCH_SIZE, step_num)
                    for k, v in div_scores(train_scores, cfg.train.GRAD_ACCUM_STEPS).items():
                        writer.add_scalar(metric2name[k] + "/Train", v, step_num)
                    train_loss = 0.0
                    train_scores = {"acc1": 0, "acc3": 0, "mrr": 0}
                    grad_accum_step_cnt = 0
                    step_num += 1
                    steps_since_last_eval += 1
                    save_checkpoint_step_cnt += 1
                    progress_bar.update(1)

                if steps_since_last_eval == cfg.train.STEPS_BETWEEN_EVAL: # add 0-th step
                    model.eval()
                    del batch

                    for i, dl in zip(["", "2"], (val_dl, val2_dl)):
                        model.eval()
                        val_loss = 0.0
                        val_scores = {"acc1": 0, "acc3": 0, "mrr": 0}
                        with torch.no_grad():
                            for batch in tqdm(dl, total=len(dl), leave=False, desc=f'Valid{i}'):
                                outputs = model(to_device(batch, DEVICE))
                                loss = loss_fn(outputs, F.one_hot(batch["label"], 10).float().to(DEVICE))
                                val_loss += loss.item()
                                new_scores = eval_batch(batch["label"], outputs)
                                val_scores = sum_scores(val_scores, new_scores)
                        logger.info(f"[{step_num}]: " + f"Loss/Validation{i}={val_loss / len(dl)}")
                        writer.add_scalar(f"Loss/Validation{i}", val_loss / len(dl), step_num)
                        for k, v in div_scores(val_scores, len(dl)).items():
                            writer.add_scalar(metric2name[k] + f"/Validation{i}", v, step_num)
                            logging.info(f"[{step_num}]: " + metric2name[k] + f"/Validation{i}={v}")

                    model.train()
                    steps_since_last_eval = 0
                if save_checkpoint_step_cnt == cfg.train.SAVE_CHECKPOINT_STEPS:
                    save_checkpoint_step_cnt = 0
                    p = Path(cfg.SAVE_CHECKPOINT_PATH) / f"step-{step_num}.pt"
                    logging.info(f"[{epoch_num}:{step_num}] Saving checkpoint to \"{str(p)}\"")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict()
                    }, p)

    if cfg.DO_TEST:
        predictions = []

        with torch.no_grad():
            for (i, batch) in enumerate(tqdm(test_dl)):
                preds = model(to_device(batch, DEVICE))[0].cpu().numpy()
                row = test_df.iloc[i]
                predictions.append({row[f"image{j}"]: preds[j] for j in range(10)})
        print("Test1: ", evaluate(
            test_df.iloc[:, 2:-1].values,
            test_df["label"].values.reshape(-1, 1),
            predictions,
        ))

        predictions = []
        csv = []
        with torch.no_grad():
            for (i, batch) in enumerate(tqdm(test2_dl)):
                preds = model(to_device(batch, DEVICE))[0].cpu().numpy()
                row = test2_df.iloc[i]
                predictions.append({row[f"image{j}"]: preds[j] for j in range(10)})

                csv.append([key for key,val in sorted(predictions[0].items(), key=lambda x: x[1], reverse=True)])
        print("Test2: ", evaluate(
            test2_df.iloc[:, 2:-1].values,
            test2_df["label"].values.reshape(-1, 1),
            predictions,
        ))
        pd.DataFrame(csv).to_csv(f'{HydraConfig.get().runtime.output_dir}/predictions.txt', sep='\t', header=False,)


if __name__ == '__main__':
    run()

# %%
