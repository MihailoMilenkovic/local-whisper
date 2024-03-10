from dataclasses import dataclass
from typing import Any, Dict, Union, List

import argparse

import datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from evaluate import load
import torch

# code mostly copied from https://huggingface.co/openai/whisper-medium


def map_to_pred(batch):
    input_features = batch["test_input_features"]
    with torch.no_grad():
        predicted_ids = glob_model.generate(input_features.to("cuda"))[0]
    transcription = glob_processor.decode(predicted_ids)
    batch["prediction"] = glob_processor.tokenizer._normalize(transcription)
    return batch


def get_eval_info(model, processor, dataset):
    global glob_model
    global glob_processor
    glob_model = model
    glob_processor = processor
    # TODO: run model on dataset and get error rate
    result = dataset.map(map_to_pred)
    wer = load("wer")
    # if i % 100 == 0:
    #     print(
    #         f"PREDICTION:{result['prediction']}, REFERENCE:{result['test_reference']}"
    #     )
    return 100 * wer.compute(
        references=result["test_reference"], predictions=result["prediction"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt_location",
        type=str,
        required=True,
        help="Checkpoint of whisper model to train",
    )
    parser.add_argument(
        "--lora_ckpt_location",
        type=str,
        help="Checkpoint of finetuned whisper model to evaluate",
    )
    parser.add_argument(
        "--dataset_location",
        type=str,
        required=True,
        help="Location from which to load data",
    )

    args = parser.parse_args()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_ckpt_location, device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(args.model_ckpt_location)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="sr", task="transcribe"
    )
    model = model.to("cuda")
    if args.lora_ckpt_location:
        model = PeftModel.from_pretrained(model, args.lora_ckpt_location)
    dataset = datasets.load_from_disk(args.dataset_location)
    dataset.set_format(type="torch", columns=["input_features"])
    res = get_eval_info(model, processor, dataset)
    print(res)
