import argparse

from transformers import WhisperProcessor
import datasets
from datasets import Audio

from utils import transliterate_cir2lat

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
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="sinhalese", task="transcribe"
)


def is_audio_in_length_range(length):
    return length < MAX_INPUT_LENGTH


def prepare_dataset(example):
    audio = example["audio"]

    example["sentence"] = transliterate_cir2lat(example["sentence"])

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


def create_dataset():
    dataset_list = []
    for config in dataset_configs_to_use:
        for language in config["languages"]:
            print(f"loading {language} for dataset {config['dataset_name']}")
            curr_dataset = datasets.load_dataset(
                config["dataset_name"],
                language,
                split="train+validation+test",
                trust_remote_code=True,
            )
            print("initial dataset:", curr_dataset)
            curr_dataset = datasets.Dataset.from_dict(
                {
                    "sentence": curr_dataset[config["text_column"]],
                    "audio": curr_dataset[config["audio_column"]],
                }
            )
            print("dataset with filtered cols:", curr_dataset)
            sampling_rate = processor.feature_extractor.sampling_rate
            curr_dataset = curr_dataset.cast_column(
                "audio", Audio(sampling_rate=sampling_rate)
            )
            curr_dataset = curr_dataset.map(
                prepare_dataset,
                num_proc=1,
            )
            print("prepared dataset:", curr_dataset)
            curr_dataset = curr_dataset.filter(
                is_audio_in_length_range, input_columns=["input_length"]
            )
            print("length filtered dataset:", curr_dataset)
            dataset_list.append(curr_dataset)

    new_dataset = datasets.concatenate_datasets(dataset_list)
    new_dataset = new_dataset.shuffle(seed=42)
    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_save_location", type=str, required=True)
    args = parser.parse_args()

    dataset = create_dataset()
    dataset.save_to_disk(args.dataset_save_location)
