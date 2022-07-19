"""
Trains UniXcoder. Requires ./datasets/train.txt and ./datasets/validation.jsonl to be present.

Usage:
   train.py <dataset_folder> [options]

Options:
    -m <model_name>, --model_name <model_name>                      Set the name of the model to be trained [default: microsoft/unixcoder-base]
    -l <learning_rate>, --learning_rate <learning_rate>             Set the learning rate [default: 0.00002]
    --max_input_length <max_input_length>                           Set the maximum length of the input. If the input is larger than this length it is truncated
                                                                    on the left side. Shorter inputs are padded [default: 936]
    --max_output_length <max_output_length>                         Set the maximum length of the output [default: 64]
    -s <seed>, --seed <seed>                                        The seed used for randomized things [default: 42]
    --beam_size <beam_size>                                         Set the beam size for beam search [default: 3]
    -bs <batch_size>, --batch_size <batch_size>                     Set the batch size [default: 32]
    -ep <num_epochs>, --num_epochs <num_epochs>                     Set the number of epochs [default: 10]
    --gradient_accumulation_steps <gradient_accumulation_steps>     Set the number of steps to accumulate gradients [default: 1]
    --weight_decay <weight_decay>                                   Set the weight decay [default: 0.0]
    --adam_epsilon <adam_epsilon>                                   Set the epsilon for Adam optimizer [default: 0.00000001]
"""
import multiprocessing
from typing import List

import torch
from docopt import docopt
import random
import os
import numpy as np
from fuzzywuzzy import fuzz
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
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
    return os.path.join(dataset_folder, "datasets", "models")


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
        eos_id=tokenizer.sep_token_id,
    )

    return model, tokenizer


def read_train_examples(dataset_folder: str):
    train_file_path = get_train_file_path(dataset_folder)

    with open(train_file_path, "r") as f:
        examples = [
            " ".join(line.strip().split()[1:])
            for line in f
            if len(line.strip()) > 0
        ]

    return examples


def read_validation_examples(dataset_folder: str):
    validation_file_path = get_validation_file_path(dataset_folder)

    examples = []
    with open(validation_file_path, "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            example = json.loads(line)

            # replace \n with </s>, remove leading <s>, normalize spacing
            left_context = example["left_context"]
            left_context = left_context.replace("\n", " </s> ") + " </s>"
            left_context = left_context.split()[1:]
            left_context = " ".join(left_context)

            examples.append({
                "input": left_context,
                "output": example["groundTruth"],
            })

    return examples


def tokenize(item):
    i, example, max_length, tokenizer = item
    source_tokens = [x for x in tokenizer.tokenize(example) if x != '\u0120']
    source_tokens = ["<s>", "<decoder-only>", "</s>"] + source_tokens[-(max_length - 3):]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = max_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    if i < 5:
        print(f"Example {i}")
        print(f"Source Tokens:", [x.replace('\u0120', '_') for x in source_tokens])
        print(f"Source IDs: {' '.join(map(str, source_ids))}")

    return source_ids


def examples_to_features(examples: List[str], tokenizer: RobertaTokenizer, max_length: int):
    pool = multiprocessing.Pool(max(1, int(os.cpu_count() * 0.8)))

    sources = [(i, example, max_length, tokenizer) for i, example in enumerate(examples)]
    print("*** Examples ***")
    features = pool.map(
        tokenize,
        tqdm(sources, total=len(sources)),
    )

    return features


def main(
        dataset_folder: str,
        model_name: str,
        learning_rate: float,
        max_input_length: int,
        max_output_length: int,
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

    os.makedirs(os.path.join(dataset_folder, "models"), exist_ok=True)

    model, tokenizer = prepare_model(model_name, max_input_length, beam_size)

    if torch.cuda.is_available():
        print("CUDA available, using GPU")
        device = torch.device("cuda")
        model = model.to(device)
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
        model = model.to(device)

    print("Loading training examples")
    train_examples = read_train_examples(dataset_folder)
    # TODO: Windowed approach if the example is too big!!!
    train_features = examples_to_features(train_examples, tokenizer, max_input_length + max_output_length)
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.long))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size // gradient_accumulation_steps
    )

    print("Loading validation examples")
    validation_examples = read_validation_examples(dataset_folder)
    validation_features = examples_to_features([x["output"] for x in validation_examples], tokenizer, max_input_length)

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
    print("  Num examples = %d", len(train_examples))
    print("  Num Epochs = %d", num_epochs)
    print("  Batch size = %d", batch_size)
    print("  Steps per epoch = %d", len(train_examples) // batch_size)
    print("  Total optimization steps = %d", len(train_dataloader) * num_epochs)

    # this is all copied from the unixcoder code
    model.train()
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_accuracy, best_loss = 0, 0, 0, 0, 0, 1e6
    losses = []
    for epoch in range(num_epochs):

        for idx, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
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
        print("  Num examples = %d" % len(validation_examples))

        model.eval()
        all_source_ids = torch.tensor(validation_features, dtype=torch.long)
        validation_dataset = TensorDataset(all_source_ids)
        validation_sampler = SequentialSampler(validation_dataset)
        validation_dataloader = DataLoader(
            validation_dataset,
            sampler=validation_sampler,
            batch_size=batch_size
        )

        p = []
        for batch in tqdm(validation_dataloader, total=len(validation_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
            with torch.no_grad():
                preds = model(source_ids=source_ids)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    if "{" in text:
                        text = text[:text.index("{")]
                    if "</s>" in text:
                        text = text[:text.index("</s>")]
                    p.append(text)
        model.train()
        EM = 0.0
        edit_sim = 0.0
        total = len(p)
        for ref, gold in zip(p, validation_examples):
            pred = " ".join(ref.strip().split())
            gt = " ".join(gold["output"].strip().split())
            edit_sim += fuzz.ratio(pred, gt)
            if pred.split() == gt.split():
                EM += 1
        validation_accuracy = round(EM / total * 100, 2)
        print("  %s = %s " % ("Acc", str(validation_accuracy)))
        print("  %s = %s " % ("Edit sim", str(round(edit_sim / total, 2))))
        print("  " + "*" * 20)
        best_model = False
        if validation_accuracy > best_accuracy:
            print("  Best acc: %s" % validation_accuracy)
            print("  " + "*" * 20)
            best_accuracy = validation_accuracy
            best_model = True
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_name = f"epoch-{epoch}-acc-{validation_accuracy*100:.3f}"
        torch.save(model_to_save.state_dict(), get_model_file_path(dataset_folder, model_name))
        if best_model:
            with open(get_model_file_path(dataset_folder, "best_model"), "w") as f:
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
        int(args['--beam_size']),
        int(args['--batch_size']),
        int(args['--num_epochs']),
        int(args['--gradient_accumulation_steps']),
        float(args['--weight_decay']),
        float(args['--adam_epsilon']),
        int(args['--seed']),
    )
