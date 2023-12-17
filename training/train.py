import argparse

import transformers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, required=True, help="Type of whisper model to train"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Number of epochs to train model for",
    )
    args = parser.parse_args()
