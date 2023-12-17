import argparse

import datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from evaluate import load
import torch

# code mostly copied from https://huggingface.co/openai/whisper-medium


def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["reference"] = processor.tokenizer._normalize(batch["text"])

    with torch.no_grad():
        predicted_ids = glob_model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch


def get_eval_info(model, dataset):
    global glob_model
    glob_model = model
    # TODO: run model on dataset and get error rate
    result = dataset.map(map_to_pred)
    wer = load("wer")
    return 100 * wer.compute(
        references=result["reference"], predictions=result["prediction"]
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
        args.model_ckpt_location, load_in_8bit=True, device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(args.model_ckpt_location)
    if args.lora_ckpt_location:
        model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
    dataset = datasets.load_from_disk(args.dataset_location, split="test")

    res = get_eval_info(model, dataset)
    print(res)
