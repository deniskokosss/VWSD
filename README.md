# VWSD
This is CMC MSU repo for SemEval 2023 Visual Word Sense Disambiguation

## Usefull links

Competition main page https://raganato.github.io/vwsd/ \
Our local leaderboard (see dev-test split below) https://docs.google.com/spreadsheets/d/18IWTnU6_e1_qZe_VTzCGC4yN9XdHFFH0qv2lakZdP4k/edit?usp=sharing

## Preparation steps
### Expected folder structure
/data \
/--- train_v1 \
/------ train.data.v1.txt \
/------ train.gold.v1.txt \
/------ train_images_v1 \
/--- trial_v1 \
/------ ...

/scripts \
/--- * runable .py scripts *

/notebooks \
/--- * .ipynb notebooks

/src \
/--- * .py modules *

### Generating dev-test split and submission
```
cd scripts
python generate_dev_test_split.py
python generate_sample_submission.py
python evaluate ../data/train_v1/sample_submission.csv
```
- data/train_v1/split_test.txt - Zero-based **test** indexes in train.data.v1.txt and train.gold.v1.txt
- data/train_v1/split_train.txt - Zero-based **train** indexes in train.data.v1.txt and train.gold.v1.txt
- data/train_v1/sample_submission.csv - Sample submission on test set

