import os
import argparse
from typing import Optional
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
import datasets
from datasets import Audio

from utils import transliterate_cir2lat, transliterate_lat2cir

MAX_INPUT_LENGTH = 30.0
# include serbian, bosnian and croatian
dataset_configs_to_use = [
    {
        "dataset_name": "mozilla-foundation/common_voice_16_0",
        "languages": ["sr"],
        "audio_column": "audio",
        "text_column": "sentence",
    },
    # {
    #     "dataset_name": "facebook/voxpopuli",
    #     "languages": ["hr"],
    #     "audio_column": "audio",
    #     "text_column": "sentence",
    # },
    # {
    #     "dataset_name": "google/fleurs",
    #     "languages": ["Serbian", "Croatian", "Bosnian"],
    #     "audio_column": "audio",
    #     "text_column": "sentence",
    # },
]
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="serbian", task="transcribe"
)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="serbian", task="transcribe"
)


def is_audio_in_length_range(length):
    return length < MAX_INPUT_LENGTH


def prepare_text(batch, use_cyrilic: Optional[bool] = False):

    batch["sentence"] = (
        transliterate_cir2lat(batch["sentence"])
        if use_cyrilic
        else transliterate_cir2lat(batch["sentence"])
    )

    return batch


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    # NOTE: alternative way to create input features here
    # batch["test_input_features"] = processor(
    #     audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    # ).input_features
    batch["test_reference"] = processor.tokenizer._normalize(batch["sentence"])
    return batch


def create_dataset(use_cyrilic: Optional[bool] = True, split: str = "train"):
    dataset_list = []
    for config in dataset_configs_to_use:
        for language in config["languages"]:
            print(
                f"loading {language} for split {split} dataset {config['dataset_name']}"
            )
            curr_dataset = datasets.load_dataset(
                config["dataset_name"],
                language,
                split=split,
                trust_remote_code=True,
            )
            print("initial dataset:", curr_dataset)
            curr_dataset = datasets.Dataset.from_dict(
                {
                    "sentence": curr_dataset[config["text_column"]],
                    "audio": curr_dataset[config["audio_column"]],
                }
            )
            curr_dataset = curr_dataset.select(range(10))
            sampling_rate = processor.feature_extractor.sampling_rate
            print("new sampling rate for audio is", sampling_rate)
            curr_dataset = curr_dataset.cast_column(
                "audio", Audio(sampling_rate=sampling_rate)
            )
            curr_dataset = curr_dataset.map(
                prepare_text, fn_kwargs={"use_cyrilic": use_cyrilic}
            )
            curr_dataset = curr_dataset.map(
                prepare_dataset,
                remove_columns=curr_dataset.column_names,
                num_proc=4,
            )
            print("prepared dataset:", curr_dataset)
            # curr_dataset = curr_dataset.filter(
            #     is_audio_in_length_range, input_columns=["input_length"]
            # )
            # print("length filtered dataset:", curr_dataset)

            dataset_list.append(curr_dataset)

    new_dataset = datasets.concatenate_datasets(dataset_list)
    new_dataset = new_dataset.shuffle(seed=42)
    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_save_location", type=str, required=True)
    parser.add_argument("--use_cyrilic", action="store_true")
    args = parser.parse_args()

    splits = ["train", "validation"]
    for split in splits:
        dataset = create_dataset(use_cyrilic=args.use_cyrilic, split=split)
        split_save_location = os.path.join(args.dataset_save_location, split)
        dataset.save_to_disk(split_save_location)
