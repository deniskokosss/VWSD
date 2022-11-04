from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    PART = 'train'
    PATH = Path('../data').resolve() / f"{PART}_v1"
    
    data = pd.read_csv(PATH / f"{PART}.data.v1.txt", sep='\t', header=None)
    train, test = train_test_split(data, test_size=0.25, random_state=123)

    assert sum(train.index) == 62181533
    assert sum(test.index) == 20617613

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





    