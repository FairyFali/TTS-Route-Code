"""Step 4: train the proxy reward model.

Reproduced from `notebooks/reward_modeling.py` in microsoft/best-route-llm, which is
itself HuggingFace's TRL reward-modelling example (Apache-2.0). Behaviour is unchanged;
the only edits are (a) tolerating a "file://" prefix on --data_path, and (b) logging the
pair counts so a bad dataset is obvious before a multi-hour run.

Standard Bradley-Terry pairwise objective over a sequence-classification head with
num_labels=1: the scalar assigned to `chosen` is trained to exceed that of `rejected`.
"""

import logging
import os
import warnings

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import (ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map,
                 get_peft_config, get_quantization_config)

logging.basicConfig(level=logging.INFO)
tqdm.pandas()

if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    config, model_config, extra_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    if extra_args:
        assert extra_args[0] == "--data_path", "The first extra argument should be --data_path"
        data_path = extra_args[1]
    else:
        raise ValueError("Missing data path.")
    data_path = data_path[len("file://"):] if data_path.startswith("file://") else data_path
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    torch_dtype = (model_config.torch_dtype
                   if model_config.torch_dtype in ["auto", None]
                   else getattr(torch, model_config.torch_dtype))
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(revision=model_config.model_revision,
                        device_map=get_kbit_device_map() if quantization_config is not None else None,
                        quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
        use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1,
        trust_remote_code=model_config.trust_remote_code, **model_kwargs)

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn("Pass --lora_task_type SEQ_CLS when using PEFT here.")

    raw_datasets = load_from_disk(data_path)

    def preprocess_function(examples):
        new_examples = {"input_ids_chosen": [], "attention_mask_chosen": [],
                        "input_ids_rejected": [], "attention_mask_rejected": []}
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tc = tokenizer(chosen)
            tr = tokenizer(rejected)
            new_examples["input_ids_chosen"].append(tc["input_ids"])
            new_examples["attention_mask_chosen"].append(tc["attention_mask"])
            new_examples["input_ids_rejected"].append(tr["input_ids"])
            new_examples["attention_mask_rejected"].append(tr["attention_mask"])
        return new_examples

    before = {k: len(v) for k, v in raw_datasets.items()}
    raw_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=4)
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= config.max_length
        and len(x["input_ids_rejected"]) <= config.max_length)
    after = {k: len(v) for k, v in raw_datasets.items()}
    # A large drop here means max_length is truncating away most of the corpus -- that is
    # silent in the original script and produces a model trained on short answers only.
    logging.info("pairs before length filter: %s", before)
    logging.info("pairs after  length filter (max_length=%d): %s", config.max_length, after)

    trainer = RewardTrainer(model=model, tokenizer=tokenizer, args=config,
                            train_dataset=raw_datasets["train"],
                            eval_dataset=raw_datasets["validation"],
                            peft_config=get_peft_config(model_config))
    trainer.train()
    best = os.path.join(config.output_dir, "checkpoint-best")
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    logging.info(metrics)
