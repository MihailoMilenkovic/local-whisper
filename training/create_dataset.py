import os
import argparse
from typing import Optional, Tuple
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

def load_dataset_from_folder(input_folder: str, split_fraction: float = 0.8) -> dict:
    # List all files in the directory
    files = os.listdir(input_folder)
    
    # Filter .txt and .mp3 files
    txt_files = sorted([f for f in files if f.endswith('.txt')])
    mp3_files = sorted([f for f in files if f.endswith('.mp3')])

    # Ensure that there is a matching .txt file for each .mp3 file
    assert len(txt_files) == len(mp3_files), "Mismatch between text and audio files."
    
    # Prepare lists for the columns 'sentence' and 'audio'
    sentences = []
    audio_paths = []

    for txt_file, mp3_file in zip(txt_files, mp3_files):
        if os.path.splitext(txt_file)[0] == os.path.splitext(mp3_file)[0]:
            # Read text from the .txt file
            with open(os.path.join(input_folder, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
            # Add the text and corresponding audio file path to their respective lists
            sentences.append(text)
            audio_paths.append(os.path.join(input_folder, mp3_file))
        else:
            raise ValueError(f"Mismatch between file names: {txt_file} and {mp3_file}")

    # Create the dataset dictionary with 'sentence' and 'audio' keys
    data_dict = {
        "sentence": sentences,
        "audio": audio_paths
    }

    # Load the data into a Hugging Face Dataset
    dataset = datasets.Dataset.from_dict(data_dict)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio())

    # Split the dataset into train and test sets
    train_test_data = dataset.train_test_split(test_size=1 - split_fraction)

    return train_test_data


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
        transliterate_lat2cir(batch["sentence"])
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
    print("BATCH LABELS",batch["labels"])
    # NOTE: alternative way to create input features here
    # batch["test_input_features"] = processor(
    #     audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    # ).input_features
    # batch["test_reference"] = processor.tokenizer._normalize(batch["sentence"])
    batch["test_reference"] = batch["sentence"]
    return batch

def create_custom_dataset(input_folder:str)->Tuple[datasets.Dataset,datasets.Dataset]:
    dataset=load_dataset_from_folder(input_folder=input_folder,split_fraction=0.8)
    train_dataset=dataset["train"]
    test_dataset=dataset["test"]
    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    test_dataset = test_dataset.map(
        prepare_dataset,
        remove_columns=test_dataset.column_names,
        num_proc=4,
    )
    return train_dataset,test_dataset

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
            # curr_dataset = curr_dataset.select(range(10))
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
    use_existing_dataset=False
    if use_existing_dataset:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_save_location", type=str, required=True)
        parser.add_argument("--use_cyrilic", action="store_true")
        args = parser.parse_args()
        print("Dataset save location:", args.dataset_save_location)
        print("Cyrilic:", args.use_cyrilic)
        splits = ["train", "validation"]
        for split in splits:
            dataset = create_dataset(use_cyrilic=args.use_cyrilic, split=split)
            split_save_location = os.path.join(args.dataset_save_location, split)
            dataset.save_to_disk(split_save_location)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_input_folder", type=str, required=True)
        parser.add_argument("--dataset_save_location", type=str, required=True)
        args=parser.parse_args()
        print("loading dataset from:",args.dataset_input_folder)
        train_save_path=os.path.join(args.dataset_save_location,"train")
        test_save_path=os.path.join(args.dataset_save_location,"validation")

        os.makedirs(train_save_path,exist_ok=True)
        os.makedirs(test_save_path,exist_ok=True)

        train_dataset,test_dataset = create_custom_dataset(input_folder=args.dataset_input_folder)
        print("saving train split of dataset to",train_save_path)
        train_dataset.save_to_disk(train_save_path)
        print("saving test split of dataset to",test_save_path)
        test_dataset.save_to_disk(test_save_path)
        print("done")
        