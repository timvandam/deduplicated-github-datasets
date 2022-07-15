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
import csv


def main(dataset_folder: str, test_pct: float, train_pct: float, validation_pct: float, seed: int):
    if abs(1 - (test_pct + train_pct + validation_pct)) > 0.00001:
        print(f"Train + test + validation must be equal to 1, but received {test_pct + train_pct + validation_pct}")
        exit(1)

    random.seed(seed)

    files_by_repo: Dict[str, List[str]] = {}

    for file_name in os.listdir(os.path.join(dataset_folder, 'repository-files')):
        if not os.path.isfile(os.path.join(dataset_folder, 'repository-files', file_name)):
            continue

        repo_name = extract_repository_name(file_name)
        if repo_name not in files_by_repo:
            files_by_repo[repo_name] = []

        files_by_repo[repo_name].append(file_name)

    choices = np.array(random.choices(['test', 'train', 'validation'], weights=[test_pct, train_pct, validation_pct],
                                      k=len(files_by_repo.keys())))

    sets = ['test', 'train', 'validation']

    for set_name in sets:
        repository_indices = np.where(choices == set_name)[0]
        repository_names = np.array(list(files_by_repo.keys()))[repository_indices]

        path = os.path.join(dataset_folder, './sets', f'{set_name}.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Repository Name', 'File Name'])
            for repo in repository_names:
                for file_name in files_by_repo[repo]:
                    writer.writerow([fix_repository_name(repo), file_name])


def extract_repository_name(file_name: str) -> str:
    return file_name[:file_name.rindex("-")]


def fix_repository_name(underscored_repository_name: str):
    first_underscore = underscored_repository_name.index("_")
    if first_underscore == -1:
        raise Exception(f"Full repository name should have an underscore. Invalid: '{underscored_repository_name}'")
    return underscored_repository_name[:first_underscore] + '/' + underscored_repository_name[first_underscore + 1:]

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args['DATASET_FOLDER'], float(args['TEST_PCT']), float(args['TRAIN_PCT']), float(args['VALIDATION_PCT']), int(args['SEED'] or 42))
