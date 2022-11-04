import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import top_k_accuracy_score


def get_golden_index(df: pd.DataFrame, gold: pd.DataFrame) -> np.array:
    ans = np.argwhere(df.values == gold.values)[:, 1]
    return ans

def mrr(gold: np.array, preds: np.array):
    ranks = preds.argsort(1)[:, ::-1]
    ans = np.argwhere(ranks == gold[:, None])
    return (1 / (1 + ans)).mean()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Please provide filepath')

    filepath = Path(sys.argv[1]).resolve()
    print(f"Evaluating {filepath} ...")
    preds = pd.read_csv(filepath, sep='\t', header=None)

    PART = 'train'
    PATH = Path('../data').resolve() / f"{PART}_v1"
    data = pd.read_csv(PATH / f"{PART}.data.v1.txt", sep='\t', header=None)
    gold = pd.read_csv(PATH / f"{PART}.gold.v1.txt", sep='\t', header=None)
    test_idx = pd.read_csv(PATH / f"split_test.txt", sep='\t', header=None).T.values[0]

    gold = get_golden_index(data.loc[test_idx, :], gold.loc[test_idx, :])

    print(f"Accuracy@Top1 {top_k_accuracy_score(gold, preds.values, k=1):.4f}")
    print(f"Accuracy@Top3 {top_k_accuracy_score(gold, preds.values, k=3):.4f}")
    print(f"Mean reciprocal rank {mrr(gold, preds.values):.4f}")


