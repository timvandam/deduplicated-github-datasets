"""
Trains UniXcoder. Requires ./datasets/train.txt and ./datasets/validation.jsonl to be present.

Usage:
   train.py <dataset_folder> [options]

Options:
    -m <model_name>, --model_name <model_name>                      Set the name of the model to be trained
                                                                    [default: microsoft/unixcoder-base]
    -l <learning_rate>, --learning_rate <learning_rate>             Set the learning rate [default: 0.00002]
    --max_input_length <max_input_length>                           Set the maximum length of the input. If the
                                                                    validation input is larger than this length it is
                                                                    truncated from the left. Train data uses a windowed
                                                                    approach instead (see window_offset)  [default: 936]
    --max_output_length <max_output_length>                         Set the maximum length of the output [default: 64]
    --window_offset <window_offset>                                 Set the window offset used for windowing train
                                                                    inputs larger than the max allowed length.
                                                                    The next window always starts at
                                                                    *current_window_end - window_offset* [default: 100]
    -s <seed>, --seed <seed>                                        The seed used for randomized things [default: 42]
    --beam_size <beam_size>                                         Set the beam size for beam search [default: 3]
    -bs <batch_size>, --batch_size <batch_size>                     Set the batch size [default: 8]
    -ep <num_epochs>, --num_epochs <num_epochs>                     Set the number of epochs [default: 10]
    --gradient_accumulation_steps <gradient_accumulation_steps>     Set the number of steps to accumulate gradients
                                                                    [default: 1]
    --weight_decay <weight_decay>                                   Set the weight decay [default: 0.0]
    --adam_epsilon <adam_epsilon>                                   Set the epsilon for Adam optimizer
                                                                    [default: 0.00000001]
"""
from multiprocessing import Pool
from typing import List
import torch
from docopt import docopt
import random
import os
import numpy as np
from fuzzywuzzy import fuzz
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from Seq2Seq import Seq2Seq
import json
import sys
from functools import lru_cache


def cache(f):
    return lru_cache(maxsize=None)(f)


def log(*args):
    print(*args, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_train_file_path(dataset_folder: str):
    return os.path.join(dataset_folder, "datasets", "train.txt")


def get_validation_file_path(dataset_folder: str):
    return os.path.join(dataset_folder, "datasets", "validation.jsonl")


def get_model_folder_path(dataset_folder: str):
    return os.path.join(dataset_folder, "models")


def get_model_file_path(dataset_folder: str, model_name: str):
    return os.path.join(get_model_folder_path(dataset_folder), model_name + ".bin")


def verify_files_exist(dataset_folder: str):
    if not os.path.exists(dataset_folder):
        raise Exception("Dataset folder does not exist")

    if not os.path.exists(get_train_file_path(dataset_folder)):
        raise Exception("Train file does not exist")

    if not os.path.exists(get_validation_file_path(dataset_folder)):
        raise Exception("Validation file does not exist")


def prepare_model(model_name: str, max_input_length: int, beam_size: int):
    config = RobertaConfig.from_pretrained(model_name)
    config.is_decoder = True
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # confirm assumptions that are made in createTrainSet.ts
    if tokenizer.bos_token != '<s>':
        raise Exception("Tokenizer bos_token is not <s>")

    if tokenizer.cls_token != '<s>':
        raise Exception("Tokenizer cls_token is not <s>")

    if tokenizer.sep_token != '</s>':
        raise Exception("Tokenizer sep_token is not </s>")

    if tokenizer.eos_token != '</s>':
        raise Exception("Tokenizer eos_token is not </s>")

    encoder = RobertaModel.from_pretrained(model_name, config=config)
    decoder = encoder
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        config=config,
        beam_size=beam_size,
        max_length=max_input_length,
        sos_id=tokenizer.cls_token_id,
        eos_id=[tokenizer.sep_token_id],
    )

    return model, tokenizer


class AbstractIterableDataset(IterableDataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.line_count = -1

    def _extract_inputs(self, line: str):
        """
        Extracts inputs from a line.
        Should return an iterable of inputs.
        """
        raise NotImplementedError()

    def _process_input(self, data):
        """
        Converts one input into one output (eg tokens to token ids)
        """
        raise NotImplementedError()

    def _get_input_count(self, line: str):
        return len(self._extract_inputs(line))

    @cache
    def __len__(self):
        """
        :return: The amount of inputs
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if len(line.strip()) > 0)

        with open(self.file_path, 'r', encoding='utf-8') as f, Pool(os.cpu_count()) as pool:
            amounts = sum(pool.imap_unordered(
                self._get_input_count,
                tqdm(f, total=total_lines, file=sys.stdout),
                chunksize=500
            ))
            return amounts

    def _extract_processed_inputs(self, line: str):
        inputs = self._extract_inputs(line)
        return [self._process_input(x) for x in inputs]

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f, Pool(os.cpu_count()) as pool:
            yield from pool.imap_unordered(
                self._extract_processed_inputs,
                f,
                chunksize=10  # low chunk size because getting token ids is expensive and we want them fast
            )


def tokens_to_token_ids(tokenizer: RobertaTokenizer, max_length: int, tokens: List[str]):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = max_length - len(token_ids)
    token_ids += [tokenizer.pad_token_id] * padding_length
    return token_ids


class TrainDataset(AbstractIterableDataset):
    def __init__(self, train_file_path: str, max_length: int, tokenizer: RobertaTokenizer, window_offset: int):
        super().__init__(train_file_path)

        if window_offset < 0:
            raise Exception("Window offset must be >= 0")

        if window_offset >= max_length:
            raise Exception("Window offset must be < max_length")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.window_offset = window_offset

    def _extract_inputs(self, line: str):
        result = []

        data = " ".join(line.strip().split()[1:])
        tokens = [token for token in self.tokenizer.tokenize(data) if token != '\u0120']

        if len(tokens) < self.max_length - 3:
            # yield ["<s>", "<decoder-only>", "</s>"] + tokens
            result.append(["<s>", "<decoder-only>", "</s>"] + tokens)
            return result

        for window in np.lib.stride_tricks.sliding_window_view(
                tokens,
                window_shape=self.max_length - 3
        )[::self.max_length - 3 - self.window_offset]:
            # yield ["<s>", "<decoder-only>", "</s>"] + list(window)
            result.append(["<s>", "<decoder-only>", "</s>"] + list(window))

        return result

    def _process_input(self, tokens: List[str]):
        return tokens_to_token_ids(self.tokenizer, self.max_length, tokens)


class ValidationDataset(AbstractIterableDataset):
    def __init__(self, validation_file_path: str, max_length: int, tokenizer: RobertaTokenizer):
        super().__init__(validation_file_path)

        self.max_length = max_length
        self.tokenizer = tokenizer

    def _extract_inputs(self, line: str):
        obj = json.loads(line)

        # replace \n with </s>, normalize spacing, add </s> to end
        left_context = obj["leftContext"]
        left_context = left_context.replace("\n", " </s> ") + " </s>"
        left_context = left_context.split()
        left_context = " ".join(left_context)

        tokens = [token for token in self.tokenizer.tokenize(left_context) if token != '\u0120']

        # truncate from the left side and add prefix
        tokens = ["<s>", "<decoder-only>", "</s>"] + tokens[-(self.max_length - 3):]

        return [{
            "input": tokens,
            "output": obj["groundTruth"],
        }]

    def _process_input(self, obj):
        return tokens_to_token_ids(self.tokenizer, self.max_length, obj["input"]), obj["output"]


def main(
        dataset_folder: str,
        model_name: str,
        learning_rate: float,
        max_input_length: int,
        max_output_length: int,
        window_offset: int,
        beam_size: int,
        batch_size: int,
        num_epochs: int,
        gradient_accumulation_steps: int,
        weight_decay: float,
        adam_epsilon: float,
        seed: int
):
    set_seed(seed)
    verify_files_exist(dataset_folder)

    os.makedirs(get_model_folder_path(dataset_folder), exist_ok=True)

    model, tokenizer = prepare_model(model_name, max_input_length, beam_size)

    if torch.cuda.is_available():
        log("CUDA available, using GPU")
        device = torch.device("cuda")
        model = model.to(device)
    else:
        log("CUDA not available, using CPU")
        device = torch.device("cpu")
        model = model.to(device)

    log("Loading validation dataset")
    validation_dataset = ValidationDataset(
        validation_file_path=get_validation_file_path(dataset_folder),
        max_length=max_input_length,
        tokenizer=tokenizer
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size
    )
    log(f"Loaded validation dataset: {len(validation_dataloader)} batches")

    log("Loading train dataset")
    train_dataset = TrainDataset(
        train_file_path=get_train_file_path(dataset_folder),
        max_length=max_input_length + max_output_length,
        tokenizer=tokenizer,
        window_offset=window_offset
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // gradient_accumulation_steps
    )
    log(f"Loaded train dataset: {len(train_dataloader)} batches")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_dataloader) * num_epochs * 0.1),
        num_training_steps=len(train_dataloader) * num_epochs
    )

    log("***** Running Training *****")
    log("\tNum examples = %d" % len(train_dataset))
    log("\tBatch size = %d" % batch_size)
    log("\tNum epochs = %d" % num_epochs)
    log("\tSteps per epoch = %d" % len(train_dataloader))

    model.train()
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_accuracy, best_loss = 0, 0, 0, 0, 0, 1e6
    losses = []
    for epoch in range(num_epochs):
        for idx, batch in enumerate(train_dataloader):
            source_ids = torch.transpose(torch.stack(batch), 0, 1).to(device).contiguous()
            loss, _, _ = model(source_ids, True)

            losses.append(loss.item())
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            tr_loss += loss.item()
            if (idx + 1) % 100 == 0:
                # TODO: Add loss plot
                log("epoch %d step %d loss %f" % (epoch, idx + 1, round(np.mean(losses[-100:]), 4)))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        # Eval model with validation dataset
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        log("***** Running Validation *****")

        model.eval()

        EM = 0.0
        edit_sim = 0.0
        for batch, ground_truths in tqdm(validation_dataloader, total=len(validation_dataloader), file=sys.stdout):
            source_ids = torch.transpose(torch.stack(batch), 0, 1).to(device).contiguous()

            with torch.no_grad():
                preds = model(source_ids=source_ids)
                for gt, pred in zip(ground_truths, preds):
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    pred = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    if "</s>" in pred:
                        pred = pred[:pred.index("</s>")]

                    pred = " ".join(pred.strip().split())
                    gt = " ".join(gt.strip().split())

                    if pred == gt:
                        EM += 1

                    edit_sim += fuzz.ratio(pred, gt)

                EM /= len(preds)
                edit_sim /= len(preds)

        model.train()
        validation_accuracy = round(EM * 100, 2)
        log("  %s = %s " % ("Acc", str(validation_accuracy)))
        log("  %s = %s " % ("Edit sim", str(round(edit_sim, 2))))
        log("  " + "*" * 20)
        best_model = False
        if validation_accuracy > best_accuracy:
            log("  Best acc: %s" % validation_accuracy)
            log("  " + "*" * 20)
            best_accuracy = validation_accuracy
            best_model = True
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_name = f"epoch-{epoch}-acc-{str(validation_accuracy * 100).replace('.', '_')}"
        torch.save(model_to_save.state_dict(), get_model_file_path(dataset_folder, model_name))
        if best_model:
            with open(get_model_file_path(dataset_folder, "best_model"), "w", encoding="utf-8") as f:
                f.write(model_name)


if __name__ == '__main__':
    args = docopt(__doc__)
    log(args)
    main(
        args['<dataset_folder>'],
        args['--model_name'],
        float(args['--learning_rate']),
        int(args['--max_input_length']),
        int(args['--max_output_length']),
        int(args['--window_offset']),
        int(args['--beam_size']),
        int(args['--batch_size']),
        int(args['--num_epochs']),
        int(args['--gradient_accumulation_steps']),
        float(args['--weight_decay']),
        float(args['--adam_epsilon']),
        int(args['--seed']),
    )
