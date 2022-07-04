"""
Tokenize Python code.

Usage:
   tokenizefiles.py [options] INPUT_FOLDER OUTPUT_FOLDER

Options:
    --only-identifiers  Return only identifiers
    -h --help           Show this screen.

"""

from tokenize import tokenize, NAME, STRING
import keyword
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator, List
from docopt import docopt
from dpu_utils.utils import save_jsonl_gz
import os


def tokenize_file(input_folder: str, file: str, only_ids: bool = False) -> Iterator[str]:
    file_path = os.path.join(input_folder, file)
    tokens = []
    with open(file_path, 'rb') as f:
        for toknum, tokval, _, _, _ in tokenize(f.readline):
            if (not only_ids) or toknum in {NAME, STRING}:
                tokens.append(tokval)
    return {"filename": os.path.relpath(file_path, input_folder), "tokens": tokens}


def tokenize_all_files(batch_number: int, batch_count: int, files: List[str], input_folder: str, output_folder: str,
                       only_ids: bool = False):
    if len(files) == 0:
        return

    print(f"[WORKER{batch_number}] Tokenizing {len(files)} files")

    def all_file_tokenizer():
        for file in files:
            try:
                yield tokenize_file(input_folder, file, only_ids)
            except:
                pass

    save_jsonl_gz(all_file_tokenizer(), os.path.join(output_folder, f"batch-{batch_number}.json.gz"))
    print(f"[WORKER{batch_number}] Finished tokenizing {len(files)} files")


def main(input_folder: str, output_folder: str, only_ids: bool = False):
    os.makedirs(output_folder, exist_ok=True)
    # TODO: Get all files
    python_files = [
        file for file in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, file))
           and os.path.splitext(file)[1] == '.py'
    ]
    process_count = os.cpu_count()

    with ProcessPoolExecutor(process_count) as executor:
        print('Starting on', process_count, 'processes')
        executors = [
            executor.submit(
                tokenize_all_files,
                i + 1,
                process_count,
                python_files[i::process_count],
                input_folder,
                output_folder,
                only_ids,
            )
            for i in range(process_count)
        ]


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args['INPUT_FOLDER'], args['OUTPUT_FOLDER'], args['--only-identifiers'])
