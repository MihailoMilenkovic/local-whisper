import argparse
import os
import math
from dataclasses import dataclass, field
import tqdm.auto as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


import datasets
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    PeftModel,
)


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()
    model_size: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(
        default=32
    )  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


@dataclass
class CustomArguments:
    lora_ckpt_path: str = field(default=None)
    num_bits_for_training: int = field(default=8)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def save_tunable_parameters(model, path):
    model.save_pretrained(path)
    # saved_params = {
    #     k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    # }
    # torch.save(saved_params, path)


def main():
    finetune_args, peft_args, training_args, custom_args = HfArgumentParser(
        (
            FinetuneArguments,
            PEFTArguments,
            TrainingArguments,
            CustomArguments,
        )
    ).parse_args_into_dataclasses()

    print("Setup Data")
    dataset = datasets.load_from_disk(finetune_args.dataset_path)

    split_dataset = dataset.train_test_split(0.04)

    print("Setup Model")
    assert custom_args.num_bits_for_training in [4, 8]
    if custom_args.num_bits_for_training == 8:
        print(f"Setting up 8-bit precision training")
        model = transformers.LlamaForCausalLM.from_pretrained(
            finetune_args.model_path,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        print(f"Setting up 4-bit precision training")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = transformers.LlamaForCausalLM.from_pretrained(
            finetune_args.model_path, quantization_config=nf4_config
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=peft_args)
    model = get_peft_model(model, peft_config)
    if custom_args.lora_ckpt_path:
        print("LOADING LORA WEIGHTS FROM CHECKPOINT:", custom_args.lora_ckpt_path)
        model.load_state_dict(torch.load(custom_args.lora_ckpt_path), strict=False)
    else:
        print("STARTING FROM RANDOM LORA WEIGHTS")
    model.print_trainable_parameters()

    print("Train")
    trainer = Trainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        args=training_args,
    )
    trainer.train()
    save_tunable_parameters(model, os.path.join(training_args.output_dir, "params.p"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
