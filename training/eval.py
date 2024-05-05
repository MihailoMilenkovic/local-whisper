from dataclasses import dataclass
from typing import Any, Tuple, List, TypedDict, Dict, Optional
import argparse
import json
import os
import concurrent
from tqdm import tqdm
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

    @classmethod
    def from_json(cls, load_path: str):
        with open(load_path, "r") as f:
            data = json.load(f)
            wer_data = data["word_error_rate"]
            sample_data = data["samples"]
            samples = [
                TextComparison(prediction=s["prediction"], reference=s["reference"])
                for s in sample_data
            ]
            return cls(word_error_rate=wer_data, samples=samples)

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


def pad_input_ids(examples, tokenizer):
    print("INPUT FEATURES:", examples["input_features"])
    return tokenizer.pad(
        examples["input_features"], max_length=1024, return_tensors="pt"
    )


# Apply the padding function to the dataset


def get_prediction(batch, bsz):
    input_features = batch["input_features"].to(device)
    with torch.no_grad():
        predicted_ids = glob_model.generate(input_features)
    transcripts = glob_processor.tokenizer.batch_decode(predicted_ids)
    predictions = [glob_processor.tokenizer._normalize(t) for t in transcripts]
    return predictions


def get_eval_info(
    model, processor, dataset, per_device_batch_size: Optional[int] = 8
) -> EvalResults:
    global glob_model
    global glob_processor
    glob_model = model
    glob_processor = processor
    res = EvalResults(samples=[], word_error_rate=None)
    # dataset = dataset.map(
    #     pad_input_ids, batched=True, fn_kwargs={"tokenizer": processor.tokenizer}
    # )
    # TODO: fix padding to make larger batch work
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, shuffle=False
    )

    for batch in tqdm(data_loader, "Computing word error rate"):
        predictions = get_prediction(batch, bsz=per_device_batch_size)
        references = batch["test_reference"]
        res.samples.extend(
            (
                [
                    TextComparison(prediction=p, reference=r)
                    for p, r in zip(predictions, references)
                ]
            )
        )
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
        "--base_model_location",
        type=str,
        required=True,
        help="Location from which to load base whisper utils",
    )
    parser.add_argument(
        "--eval_save_path",
        type=str,
        default=os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "eval_results"
        ),
    )
    parser.add_argument("--per_device_batch_size", type=int, default=1)

    args = parser.parse_args()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_ckpt_location, device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(args.base_model_location)
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
    print("Model location:", args.model_ckpt_location)
    print("Model original location:", args.base_model_location)
    res = get_eval_info(model, processor, dataset, args.per_device_batch_size)
    print("Model WER:", res.word_error_rate)
    print("Saving results to:", args.eval_save_path)
    res.to_json(args.eval_save_path)
