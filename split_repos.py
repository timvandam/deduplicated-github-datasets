"""
Splits the GitHub datasets on a repo-by-repo basis

Usage:
   split_repos.py DATASET_FOLDER TEST_PCT TRAIN_PCT VALIDATION_PCT [SEED]
"""

from typing import Set, List, Dict
from docopt import docopt
import os
import json
import random
import numpy as np


def main(dataset_folder: str, test_pct: float, train_pct: float, validation_pct: float, seed: int):
    if test_pct + train_pct + validation_pct != 1:
        print(f"Train + test + validation must be equal to 1, but received {test_pct + train_pct + validation_pct}")
        exit(1)

    random.seed(seed)

    files_by_repo: Dict[str, List[str]] = {}

    for file_name in os.listdir(dataset_folder):
        if not os.path.isfile(os.path.join(dataset_folder, file_name)):
            continue

        repo_name = extract_repository_name(file_name)
        if repo_name not in files_by_repo:
            files_by_repo[repo_name] = []

        files_by_repo[repo_name].append(file_name)

    choices = np.array(random.choices(['test', 'train', 'validation'], weights=[test_pct, train_pct, validation_pct],
                                      k=len(files_by_repo.keys())))

    test = np.where(choices == 'test')[0]
    train = np.where(choices == 'train')[0]
    validation = np.where(choices == 'validation')[0]

    test_repos = list(files_by_repo.keys())[test]
    train_repos = list(files_by_repo.keys())[train]
    validation_repos = list(files_by_repo.keys())[validation]

    test_files = [file_name for repo in test_repos for file_name in files_by_repo[repo]]
    train_files = [file_name for repo in train_repos for file_name in files_by_repo[repo]]
    validation_files = [file_name for repo in validation_repos for file_name in files_by_repo[repo]]


def extract_repository_name(file_name: str) -> str:
    return file_name[:file_name.rindex("-")]


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args['DATASET_FOLDER'], args['TEST_PCT'], args['TRAIN_PCT'], args['VALIDATION_PCT'], int(args['SEED'] or 42))
