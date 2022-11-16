from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    PART = 'train'
    PATH = Path('../data').resolve() / f"{PART}_v1"
    
    data = pd.read_csv(PATH / f"{PART}.data.v1.txt", sep='\t', header=None)

    train, test = train_test_split(data, test_size=0.25, random_state=123)
    test = pd.concat([
        test,
        train[train[0].isin(test[0].unique())]
    ])
    train = train[~ train[0].isin(test[0].unique())]

    assert train.shape[0] + test.shape[0] == data.shape[0]
    assert not np.intersect1d(train[0], test[0])
    assert sum(train.index) == 61206091
    assert sum(test.index) == 21593055

    with open(PATH / f"split_train.txt", 'w') as f:
        text1 = [str(t) for t in train.index]
        text1.sort(key=lambda x: int(x))
        f.writelines('\n'.join(text1))

    with open(PATH / f"split_test.txt", 'w') as f:
        text2 = [str(t) for t in test.index]
        text2.sort(key=lambda x: int(x))
        f.writelines('\n'.join(text2))

    if set(text1 + text2) != set(map(str, range(data.shape[0]))):
        raise ValueError("Something went wrong while splitting")





    