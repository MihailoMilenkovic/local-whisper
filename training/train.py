from typing import Any, Dict, List, Union
import argparse
import os
import math
from dataclasses import dataclass, field
import tqdm.auto as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial

import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()
    model_size: str = field()
    use_peft: bool = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    mapping_hidden_dim: int = field(default=1024)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def save_tunable_parameters(model, path):
    model.save_pretrained(path)
    # saved_params = {
    #     k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    # }
    # torch.save(saved_params, path)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i]
        for i in range(len(pred_str_norm))
        if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


def get_peft_config(peft_args: PEFTArguments):
    assert peft_args.peft_mode == "lora"
    peft_config = LoraConfig(
        task_type=TaskType.TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    return peft_config


# TODO: check model size should change here or if this shouldn't be global...
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="serbian",
    task="transcribe",
)
metric = evaluate.load("wer")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
normalizer = BasicTextNormalizer()


def main():
    finetune_args, peft_args, training_args = HfArgumentParser(
        (
            FinetuneArguments,
            PEFTArguments,
            TrainingArguments,
        )
    ).parse_args_into_dataclasses()

    print("Setup Data")
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    # dataset = dataset.train_test_split(0.04)

    print("Load model")
    model = WhisperForConditionalGeneration.from_pretrained(
        finetune_args.model_path, load_in_8_bit=True, device_map="auto"
    )
    if finetune_args.use_peft:
        print("Setup peft")
        peft_config = get_peft_config(peft_args=peft_args)
        model = get_peft_model(model, peft_config)

    print("Setup model for training")
    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate, language="serbian", task="transcribe", use_cache=True
    )
    # TODO: add lora correction option to training here

    print("Setup training args")
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"whisper-{finetune_args.model_size}-serbian",  # name on the HF Hub (TODO: rename)
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
        gradient_checkpointing=True,
        fp16=True,  # TODO: change to  8/4 bit training later
        fp16_full_eval=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["wandb"],  # ["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # train model
    trainer.train()


if __name__ == "__main__":
    main()
