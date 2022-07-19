"""
Splits the GitHub datasets on a repo-by-repo basis

Usage:
   split_repos.py DATASET_FOLDER TEST_PCT TRAIN_PCT VALIDATION_PCT [SEED]
"""

from typing import Set, List, Dict, TypeVar
from docopt import docopt
import os
import json
import random
import numpy as np
import csv

# pick `k` items from a numpy array randomly and remove them
def pick_random(l, k):
    indices = random.sample(range(len(l)), k=k)
    picked = l[indices]
    return picked, np.delete(l, indices, axis=0)


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

    choices = np.array(list(files_by_repo.keys()))

    print(f"Splitting {len(choices)} repositories according to a {test_pct}/{train_pct}/{validation_pct} split")

    test_repo_count = int(test_pct * len(choices))
    validation_repo_count = int(validation_pct * len(choices))
    train_repo_count = len(choices) - test_repo_count - validation_repo_count

    set_repositories = {}
    set_repositories["test"], choices = pick_random(choices, test_repo_count)
    set_repositories["validation"], choices = pick_random(choices, validation_repo_count)
    set_repositories["train"], choices = pick_random(choices, train_repo_count)

    if len(choices) > 0:
        print(f"{len(choices)} repositories were not used. This should be 0 after creating the sets.")
        exit(1)

    for set_name, repository_names in set_repositories.items():
        path = os.path.join(dataset_folder, './sets', f'{set_name}.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Repository Name', 'File Name'])
            for repo in repository_names:
                for file_name in files_by_repo[repo]:
                    writer.writerow([fix_repository_name(repo), file_name])
        print(f"Created {set_name}.csv with {len(repository_names)} repositories")


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
