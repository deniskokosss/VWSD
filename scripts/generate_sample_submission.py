from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    PATH = Path('../data/train_v1').resolve()

    test_idx = pd.read_csv(PATH / f"split_test.txt", sep='\t', header=None).T.values[0]

    np.random.seed(42)
    rnd = np.random.random(size=(test_idx.shape[0], 10))
    rnd = rnd / rnd.sum(1)[:, None]

    pd.DataFrame(rnd).to_csv(PATH / "sample_submission.csv", sep='\t', index=False, header=False)

