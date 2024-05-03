from dataclasses import dataclass
from typing import Any, Tuple, List, TypedDict, Dict
import argparse
import json
import os

import datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from evaluate import load
import torch

# code mostly copied from https://huggingface.co/openai/whisper-medium

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextComparison(TypedDict):
    prediction: str
    reference: str


@dataclass
class EvalResults:
    word_error_rate: float
    samples: List[TextComparison]

    def to_json(self, save_path: str):
        with open(save_path, "w") as f:
            res_obj = {"word_error_rate": self.word_error_rate, "samples": self.samples}
            json.dump(res_obj, f)

    def compute_wer(self):
        wer = load("wer")
        wer_res = 100 * wer.compute(
            references=[s["reference"] for s in self.samples],
            predictions=[s["prediction"] for s in self.samples],
        )
        self.word_error_rate = wer_res
        return wer_res


def get_prediction(entry):
    input_features = entry["input_features"].unsqueeze(0)
    with torch.no_grad():
        predicted_ids = glob_model.generate(input_features.to(device))[0]
    transcription = glob_processor.decode(predicted_ids)
    prediction = glob_processor.tokenizer._normalize(transcription)
    return prediction


def get_eval_info(model, processor, dataset) -> EvalResults:
    global glob_model
    global glob_processor
    glob_model = model
    glob_processor = processor
    # TODO: run model on dataset and get error rate
    res = EvalResults(samples=[], word_error_rate=None)
    for entry in dataset:
        prediction = get_prediction(entry)
        reference = entry["test_reference"]
        res.samples.append(TextComparison(prediction=prediction, reference=reference))
    res.compute_wer()
    return res


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
    parser.add_argument(
        "--eval_save_path",
        type=str,
        default=os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "eval_results"
        ),
    )

    args = parser.parse_args()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_ckpt_location, device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(args.model_ckpt_location)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="sr", task="transcribe"
    )
    model = model.to(device)
    if args.lora_ckpt_location:
        model = PeftModel.from_pretrained(model, args.lora_ckpt_location)
    dataset = datasets.load_from_disk(args.dataset_location)
    dataset.set_format(
        type="torch",
        columns=["input_features"],
        output_all_columns=True,
    )
    res = get_eval_info(model, processor, dataset)
    print("Model WER:", res.word_error_rate)
    res.to_json(args.eval_save_path)
