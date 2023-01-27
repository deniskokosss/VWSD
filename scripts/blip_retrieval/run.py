
import os
from pathlib import Path
import logging
import json
from typing import *
import time
import sys

sys.path.append('../..')
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
import torch.nn as nn
from lavis.models import load_model_and_preprocess, BlipBase, load_model
from lavis.processors import load_processor
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from src.data import CustomSplitLoader
from src.utils import evaluate, mrr
from src.blip_retrieval import BLIPRetrieval, RetrievalDataset
from src.blip_itm import ItmDataset
from torch.utils.data.dataloader import default_collate


def to_device(object, device):
    if not isinstance(object, dict):
        raise NotImplementedError("Implement other types than dict if needed!")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in object.items()}


@torch.no_grad()
def get_negatives(model, dataloader, device):
    images = []
    texts = []
    ids = []
    for (i, batch) in enumerate(dataloader):
        image_embed, text_embed = model.get_embeddings(to_device(batch, device))
        images += image_embed
        texts += text_embed
        assert i == batch['image_id']
        ids += batch['image_id']
    images = torch.stack(images)
    texts = torch.stack(texts)
    sim_matrix = torch.matmul(images, texts.T)
    sim_matrix = sim_matrix - torch.eye(sim_matrix.shape[0]).to(device)
    return {
        idx.item(): [neg0.item(), neg1.item()]
        for idx, neg0, neg1 in zip(ids, torch.argmax(sim_matrix, 0), torch.argmax(sim_matrix, 1))
    }


def hard_negatives_collate(samples, hard_negatives, dataset):
    neg_samples = []
    for idx in samples['image_id']:
        for neg_idx in hard_negatives[idx.item()]:
            neg_samples.append(dataset[neg_idx])
    neg_batch = default_collate(neg_samples)

    samples['text_input'] = samples['text_input'] + neg_batch['text_input']
    samples['image'] = torch.cat([samples['image'], neg_batch['image']])
    samples['image_id'] = torch.cat([samples['image_id'], neg_batch['image_id']])
    return samples


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

    _, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", 'base', is_eval=True
    )
    if cfg.data.IMAGES_FOLDER_STRUCT:
        folder_struct = json.load(open(cfg.data.IMAGES_FOLDER_STRUCT, 'r'))
    else:
        folder_struct = None
    if cfg.DO_TRAINING:
        train_ds = RetrievalDataset(
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

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.train.TRAIN_BATCH_SIZE, drop_last=True,
                                               num_workers=cfg.NUM_WORKERS, persistent_workers=True, shuffle=True)
        train_dl_hn = torch.utils.data.DataLoader(train_ds, batch_size=1,
                                               num_workers=cfg.NUM_WORKERS, persistent_workers=True, shuffle=False)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.train.VALIDATION_BATCH_SIZE,
                                             num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        val2_dl = torch.utils.data.DataLoader(val2_ds, batch_size=cfg.train.VALIDATION_BATCH_SIZE,
                                              num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        val5_dl = torch.utils.data.DataLoader(val5_ds, batch_size=cfg.train.VALIDATION_BATCH_SIZE,
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

    blip_model = load_model("blip_retrieval", cfg.model.BLIP_VARIANT)
    model = BLIPRetrieval(blip_model, head=cfg.model.HEAD).to(DEVICE)

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


    loss_fn = nn.CrossEntropyLoss()
    if cfg.DO_TRAINING:
        TRAIN_EFFECTIVE_BATCH_SIZE = cfg.train.GRAD_ACCUM_STEPS * cfg.train.TRAIN_BATCH_SIZE

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.LR, weight_decay=cfg.train.WEIGHT_DECAY)

        if cfg.train.NUM_EPOCHS:
            num_epochs = cfg.train.NUM_EPOCHS
        else:
            num_epochs = 1 + cfg.train.NUM_STEPS // (len(train_dl) // cfg.train.GRAD_ACCUM_STEPS)
        num_training_steps = int(num_epochs * (len(train_dl) / cfg.train.GRAD_ACCUM_STEPS))
        num_warmup_steps = int(num_training_steps * cfg.train.WARMUP_STEPS_FRAC)
        iters_per_epoch = num_training_steps / num_epochs
        if cfg.train.NUM_EVALUATIONS:
            steps_between_eval = num_training_steps // cfg.train.NUM_EVALUATIONS
        else:
            steps_between_eval = cfg.train.STEPS_BETWEEN_EVAL

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"{num_training_steps} training steps which include {num_warmup_steps} warmup ones")

        step_num = 0
        steps_since_last_eval = steps_between_eval
        grad_accum_step_cnt = 0
        save_checkpoint_step_cnt = 0

        if cfg.model.CHECKPOINT:
            p = Path(cfg.SAVE_CHECKPOINT_PATH) / f"step-{cfg.model.CHECKPOINT}.pt"
            dict_ = torch.load(p)
            model.load_state_dict(dict_['model'])
            optimizer.load_state_dict(dict_['optimizer'])
            lr_scheduler.load_state_dict(dict_['scheduler'])
            step_num = cfg.train.START_FROM
            logging.info(f"Loaded {p}")
            del dict_

        progress_bar = tqdm(range(num_training_steps))
        for epoch_num in range(num_epochs):

            if cfg.train.HARD_NEGATIVES.SCHEDULE and epoch_num in cfg.train.HARD_NEGATIVES.SCHEDULE:
                logging.info(f"Mining hard negatives at epoch {epoch_num}...")
                hard_negatives = get_negatives(model, train_dl_hn, DEVICE)

            model.train()
            train_loss = 0.0
            for batch in train_dl:
                if cfg.train.HARD_NEGATIVES.SCHEDULE:
                    batch = hard_negatives_collate(batch, hard_negatives, train_ds)
                batch = to_device(batch, DEVICE)
                batch.update(
                    {
                        "epoch": epoch_num,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": step_num,
                    }
                )
                outputs = model.blip_model(batch)
                loss = outputs['loss']
                loss.backward()
                train_loss += loss.item()
                grad_accum_step_cnt += 1

                if grad_accum_step_cnt == cfg.train.GRAD_ACCUM_STEPS:
                    writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], step_num)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    writer.add_scalar("Loss/Train", train_loss / TRAIN_EFFECTIVE_BATCH_SIZE, step_num)

                    train_loss = 0.0
                    grad_accum_step_cnt = 0
                    step_num += 1
                    steps_since_last_eval += 1
                    save_checkpoint_step_cnt += 1
                    progress_bar.update(1)

                if steps_since_last_eval >= steps_between_eval: # add 0-th step
                    model.eval()
                    del batch

                    for i, dl in zip(["", "2", "WD", "TR"], (val_dl, val2_dl, val5_dl)):
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

                d = {row[f"image{j}"]: preds[j] for j in range(10)}
                predictions.append(d)
                csv.append([key for key,val in sorted(d.items(), key=lambda x: x[1], reverse=True)])
        print("Test2: ", evaluate(
            test2_df.iloc[:, 2:-1].values,
            test2_df["label"].values.reshape(-1, 1),
            predictions,
        ))
        pd.DataFrame(csv).to_csv(f'{HydraConfig.get().runtime.output_dir}/predictions.txt', sep='\t', header=False,)


if __name__ == '__main__':
    run()

# %%
