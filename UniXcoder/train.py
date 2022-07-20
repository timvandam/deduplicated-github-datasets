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
import multiprocessing
from typing import List, Union, Literal

import torch
from docopt import docopt
import random
import os
import numpy as np
from fuzzywuzzy import fuzz
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler, IterableDataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from Seq2Seq import Seq2Seq
import json


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
        print("Dataset folder does not exist")
        exit(1)
    if not os.path.exists(get_train_file_path(dataset_folder)):
        print("Train file does not exist")
        exit(1)
    if not os.path.exists(get_validation_file_path(dataset_folder)):
        print("Validation file does not exist")
        exit(1)


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
        self.line_count = self._count_lines()

    def _extract_input(self, line: str):
        """
        Extracts the input from a line
        :param line: A single line
        :return: Some output data
        """
        raise NotImplementedError()

    def _extract_output(self, data):
        """
        Converts input into some output (eg) token ids
        :param data: Input data extracted in _extract_input
        :return: An iterable containing lists of outputs (one input can have multiple outputs)
        """
        raise NotImplementedError()

    def _count_lines(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            return sum(1 for line in f)

    def __len__(self):
        return self.line_count

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue

                data = self._extract_input(line)
                yield from self._extract_output(data)


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

    def _extract_input(self, line: str):
        return " ".join(line.strip().split()[1:])

    def _tokens_to_token_ids(self, tokens: List[str]):
        return tokens_to_token_ids(self.tokenizer, self.max_length, tokens)

    def _extract_output(self, data):
        tokens = [token for token in self.tokenizer.tokenize(data) if token != '\u0120']

        if len(tokens) < self.max_length - 3:
            yield self._tokens_to_token_ids(tokens)
            return

        for window in np.lib.stride_tricks.sliding_window_view(
            tokens,
            window_shape=self.max_length - 3
        )[::self.max_length - 3 - self.window_offset]:
            full_tokens = ["<s>", "<decoder-only>", "</s>"] + list(window)
            token_ids = self._tokens_to_token_ids(full_tokens)
            yield token_ids


class ValidationDataset(AbstractIterableDataset):
    def __init__(self, validation_file_path: str, max_length: int, tokenizer: RobertaTokenizer):
        super().__init__(validation_file_path)

        self.max_length = max_length
        self.tokenizer = tokenizer

    def _extract_input(self, line: str):
        obj = json.loads(line)

        # replace \n with </s>, normalize spacing, add </s> to end
        left_context = obj["leftContext"]
        left_context = left_context.replace("\n", " </s> ") + " </s>"
        left_context = left_context.split()
        left_context = " ".join(left_context)

        return {
            "input": left_context,
            "output": obj["groundTruth"],
        }

    def _tokens_to_token_ids(self, tokens: List[str]):
        return tokens_to_token_ids(self.tokenizer, self.max_length, tokens)

    def _extract_output(self, data):
        tokens = [token for token in self.tokenizer.tokenize(data["input"]) if token != '\u0120']

        # truncate from the left side and add prefix
        tokens = ["<s>", "<decoder-only>", "</s>"] + tokens[-(self.max_length - 3):]

        yield self._tokens_to_token_ids(tokens), data["output"]


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
        print("CUDA available, using GPU")
        device = torch.device("cuda")
        model = model.to(device)
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
        model = model.to(device)

    print("Loading training dataset")
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
    print("Loaded training dataset")

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

    print("***** Running Training *****")
    print("\tNum examples = %d" % len(train_dataset) )
    print("\tBatch size = %d" % batch_size)
    print("\tNum epochs = %d" % num_epochs)
    print("\tSteps per epoch = %d" % len(train_dataloader))

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
                print("epoch %d step %d loss %f" % (epoch, idx + 1, round(np.mean(losses[-100:]), 4)))
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

        print("***** Running Validation *****")

        model.eval()
        validation_dataset = ValidationDataset(
            validation_file_path=get_validation_file_path(dataset_folder),
            max_length=max_input_length,
            tokenizer=tokenizer
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size
        )

        EM = 0.0
        edit_sim = 0.0
        for batch, ground_truths in tqdm(validation_dataloader, total=len(validation_dataloader)):
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
        print("  %s = %s " % ("Acc", str(validation_accuracy)))
        print("  %s = %s " % ("Edit sim", str(round(edit_sim, 2))))
        print("  " + "*" * 20)
        best_model = False
        if validation_accuracy > best_accuracy:
            print("  Best acc: %s" % validation_accuracy)
            print("  " + "*" * 20)
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
    print(args)
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
