import numpy as np
from sklearn.metrics import top_k_accuracy_score


def get_golden_index(df: np.array, gold: np.array) -> np.array:
    ans = np.argwhere(df == gold)[:, 1]
    return ans

def mrr(gold: np.array, preds: np.array):
    ranks = preds.argsort(1)[:, ::-1]
    ans = np.argwhere(ranks == gold[:, None])
    return (1 / (1 + ans)).mean()

def get_metrics(gold: np.array, preds: np.array):
    return {
        'acc1': top_k_accuracy_score(gold, preds, k=1),
        'acc3': top_k_accuracy_score(gold, preds, k=3),
        'mrr': mrr(gold, preds)
    }

def evaluate(df: np.array, gold: np.array, preds: np.array):
    """
    :param df: format as in train.data.v1.txt
    :param gold: format as in train.gold.v1.txt
    :param preds: format as in sample_submission.csv
    :return: dict with metrics (keys: acc1, acc3, mrr)
    """
    assert df.shape[0] == gold.shape[0] == preds.shape[0], "shape mismatch"
    gold_numerical = get_golden_index(df, gold)
    return get_metrics(gold_numerical, preds)
