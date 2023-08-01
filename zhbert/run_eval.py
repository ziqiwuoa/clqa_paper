#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy, get_full_repo_name


logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEBUG_STEPS = 5

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--uncased",
        action="store_true",
        help="Make the training samples lowercase if the pre-trained model is uncased.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Task name for preprocessing the format of the dataset.",
    )
    parser.add_argument(
        "--max_option_count",
        type=int,
        default=4,
        help="The max number of options for all questions in the QA dataset. The questions with less"
             " options than the max number of options will be left blank.",
    )
    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    column_names = raw_datasets["validation"].column_names

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    choice_names = [f"choice{i}" for i in range(args.max_option_count)]
    context_name = "context" if args.task_name != "exams" else [f"context{i}" for i in range(args.max_option_count)]
    question_name = "question"
    label_column_name = "label" if "label" in column_names else "labels"

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        question_headers = examples[question_name]
        if args.task_name == "exams":
            # In EXAMS dataset, each choice has their own context
            context_sentences = [[examples[context][i] for context in context_name] for i in range(len(question_headers))]
        else:
            context_sentences = [[context.lower() if args.uncased else context] * args.max_option_count for context in examples[context_name]]

        if args.task_name == "race":
            q_a_sentences = [
                [
                    (f"{question.replace('_', examples[choice][i]).lower() if args.uncased else question.replace('_', examples[choice][i])}"
                     if "_" in question else
                     f"{question.lower()} {examples[choice][i].lower()}" if args.uncased else f"{question} {examples[choice][i]}")
                    for choice in choice_names
                ]
                for i, question in enumerate(question_headers)
            ]
        else:
            q_a_sentences = [
                [f"{question} {examples[choice][i]}" for choice in choice_names] for i, question in
                enumerate(question_headers)
            ]

        labels = examples[label_column_name]

        # ziqi debug
        if args.debug:
            logger.info("***** Preprocessing Data Info - RAW *****")
            for i in range(DEBUG_STEPS):
                logger.info(f"  Context: {context_sentences[i]}")
                for j in range(args.max_option_count):
                    logger.info(f"  Question & Choice [{j}]: {q_a_sentences[i][j]}")
                logger.info(f"  Label: {str(labels[i])}")
                logger.info("***********************************")

        # Flatten out
        context_sentences = list(chain(*context_sentences))
        q_a_sentences = list(chain(*q_a_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            context_sentences,
            q_a_sentences,
            truncation=True,
            max_length=args.max_length,
            padding=padding,
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i : i + args.max_option_count] for i in range(0, len(v), args.max_option_count)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names
        )

    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # Metrics
    metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
    total_steps = math.ceil(len(eval_dataloader))

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total evaluation steps = {total_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
        progress_bar.update(1)

    eval_metric = metric.compute()
    accelerator.print(f"Evaluation Accuracy: {eval_metric}")


if __name__ == "__main__":
    main()