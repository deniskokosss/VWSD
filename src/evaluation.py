# TODO: evaluate, make_submission_file, evaluate_submission_file (for evaluate.py script)
import pandas as pd
import numpy as np
from typing import *
from src.utils import evaluate as evaluate_raw
from pathlib import Path

def evaluate(
    df: pd.DataFrame,
    predictions: np.array,
) -> Dict[str, float]:
    return evaluate_raw(df.drop(["word", "context", "label"], axis=1).values, df[["label",]].values, predictions)

def make_submission_file(
    predictions: np.array,
    submission_file_path: Path,
) -> None:
    pd.DataFrame(predictions).to_csv(submission_file_path, sep='\t', index=False, header=False)