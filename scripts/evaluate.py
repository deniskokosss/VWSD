import sys
from pathlib import Path

import pandas as pd

from src.utils import evaluate

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

    metrics = evaluate(data.loc[test_idx, :].iloc[:, 2:].values, gold.loc[test_idx, :].values, preds.values)

    print(f"Accuracy@Top1 {metrics['acc1']:.4f}")
    print(f"Accuracy@Top3 {metrics['acc3']:.4f}")
    print(f"Mean reciprocal rank {metrics['mrr']:.4f}")


