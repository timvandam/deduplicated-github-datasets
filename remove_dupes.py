"""
Remove near duplicates from datasets.
Only keeps the first file in each cluster.
Looks for duplicate-files.json in the provided dataset folder.
Removes duplicate files from the repository-files folder in the provided dataset folder.

Usage:
   remove_dupes.py DATASET_FOLDER
"""

from docopt import docopt
import os
import json


def main(dataset_folder: str):
    duplicate_files_path = os.path.join(dataset_folder, 'duplicate-files.json')
    repository_files_path = os.path.join(dataset_folder, 'repository-files')
    with open(duplicate_files_path, 'r') as f:
        clusters = json.loads(f.read())
        print(f"Removing files from {len(clusters)} clusters")
        removed_count = 0
        for cluster in clusters:
            # remove all but the first file
            files_to_remove = cluster[1:]
            for file in files_to_remove:
                os.remove(os.path.join(repository_files_path, file))
            removed_count += len(files_to_remove)
        print(f"Removed {removed_count} files")


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args['DATASET_FOLDER'])
