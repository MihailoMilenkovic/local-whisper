from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

import os
import torch
import torch.nn as nn

import datasets
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training


@dataclass
class FinetuneArguments:
    train_dataset_path: str = field()
    eval_dataset_path: str = field()
    model_path: str = field()
    use_peft: bool = field(default=False)
    training_quantization_num_bits: int = field(default=16)


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    lora_alpha: float = field(default=32)
    lora_dropout: float = field(default=0.1)


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
            {"input_features": feature["input_features"]} for feature in features
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
        r=peft_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
    )
    return peft_config


def get_model_quantization_config(training_quantization_num_bits: int):
    assert training_quantization_num_bits in [16, 8, 4]
    if training_quantization_num_bits == 16:
        quant_config = None
    elif training_quantization_num_bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return quant_config


# TODO: check model size should change here or if this shouldn't be global...
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="sr",
    task="transcribe",
)
metric = evaluate.load("wer")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
normalizer = BasicTextNormalizer()


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def main():
    finetune_args, peft_args, training_args = HfArgumentParser(
        (
            FinetuneArguments,
            PEFTArguments,
            TrainingArguments,
        )
    ).parse_args_into_dataclasses()

    print("Setup Data")
    train_dataset = datasets.load_from_disk(finetune_args.train_dataset_path)
    print(f"Train dataset: {train_dataset}")
    if finetune_args.eval_dataset_path is not None:
        eval_dataset = datasets.load_from_disk(finetune_args.eval_dataset_path)

    quantization_config = get_model_quantization_config(
        finetune_args.training_quantization_num_bits
    )
    print("Load model")
    model = WhisperForConditionalGeneration.from_pretrained(
        finetune_args.model_path,
        quantization_config=quantization_config,
    )
    callbacks = []
    if finetune_args.use_peft:
        # TODO: fix peft training bugs
        # see https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="sr", task="transcribe"
        )
        model.config.suppress_tokens = []
        print("Setup peft")
        model = prepare_model_for_kbit_training(model)
        peft_config = get_peft_config(peft_args=peft_args)
        model = get_peft_model(model, peft_config)
        callbacks.append(SavePeftModelCallback)
    else:
        print("Training full model")

    print("Setup model for training")
    # disable cache during training since it's incompatible with gradient checkpointing

    print("Setup training args")
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )
    model.config.use_cache = False

    # train model
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
