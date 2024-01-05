import argparse

from transformers import WhisperProcessor
import datasets
from datasets import Audio

from utils import transliterate_cir2lat

# include serbian, bosnian and croatian
dataset_configs_to_use = [
    {
        "dataset_name": "mozilla-foundation/common_voice_13_0",
        "languages": ["sr", "bs", "hr"],
        "audio_column": "audio",
        "text_column": "sentence",
    },
    {
        "dataset_name": "facebook/voxpopuli",
        "languages": ["hr"],
        "audio_column": "audio",
        "text_column": "sentence",
    },
    {
        "dataset_name": "google/fleurs",
        "languages": ["Serbian", "Croatian", "Bosnian"],
        "audio_column": "audio",
        "text_column": "sentence",
    },
]
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="sinhalese", task="transcribe"
)


def preprocess_data(sample):
    sample["text"] = transliterate_cir2lat(sample["text"])
    return sample


def create_dataset():
    dataset_list = []
    for config in dataset_configs_to_use:
        for language in config["languages"]:
            print(f"loading {language} for dataset {config['dataset_name']}")
            data = datasets.load_dataset(
                config["dataset_name"], language, split="train+validation+test",trust_remote_code=True
            )
            data = datasets.Dataset.from_dict(
                {
                    "text": data[config["text_column"]],
                    "audio": data[config["audio_column"]],
                }
            )
            sampling_rate = processor.feature_extractor.sampling_rate()
            data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))
            new_data = data.map(
                preprocess_data,
                input_columns=[config["audio_column"], config["text_column"]],
            )
            dataset_list.append(new_data)

    new_dataset = datasets.concatenate_datasets(dataset_list)
    new_dataset = new_dataset.shuffle(seed=42)
    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_save_location", type=str, required=True)
    args = parser.parse_args()

    dataset = create_dataset()
    dataset.save_to_disk(args.dataset_save_location)
