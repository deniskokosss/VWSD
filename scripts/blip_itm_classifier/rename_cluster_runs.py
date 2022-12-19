import os
from pathlib import Path
import re
import yaml
from io import StringIO

if __name__ == '__main__':
    pth = Path('runs')
    for fname in os.listdir(pth):
        if re.fullmatch("/d+", fname):
            with open(pth / fname / '.hydra' / 'overrides.yaml') as f:
                conf = yaml.safe_load(f)
            os.rename(pth / fname, pth / "_".join(conf[:2]))
