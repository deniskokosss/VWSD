from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    PATH = Path('../data/train_v1').resolve()
    PART = 'train'

    data = pd.read_csv(PATH / f"{PART}.data.v1.txt", sep='\t', header=None)
    test_idx = pd.read_csv(PATH / f"split_test.txt", sep='\t', header=None).T.values[0]
    test = data.loc[test_idx, :]

    np.random.seed(42)

    res = []
    cols = [2+i for i in range(10)]
    for idx, row in test.iterrows():
        ans = {}
        for col in cols:
            score = np.random.random()
            ans[row.loc[col]] = score
        res.append(ans)

    with open(PATH / "sample_submission.json", 'w') as f:
        json.dump(res, f, indent=2)

