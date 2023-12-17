from datasets import load_dataset

from .utils import transliterate_cir2lat

# include serbian, bosnian and croatian
dataset_configs_to_use = [
    {
        "dataset_name": "mozilla-foundation/common_voice_13_0",
        "languages": ["sr", "bs", "hr"],
    },
    {
        "dataset_name": "facebook/voxpopuli",
        "languages": ["hr"],
    },
    {
        "dataset_name": "google/fleurs",
        "languages": ["Serbian", "Croatian", "Bosnian"],
    },
]


def get_data(split: str = None):
    data = []
    for config in dataset_configs_to_use:
        for language in config["languages"]:
            curr_data = load_dataset(config["dataset_name"], language)
            audio = None
            caption = None
            # map all text to latin for normalization
            caption = transliterate_cir2lat(caption)
            new_data = language
            data.extend(new_data)
        return data
