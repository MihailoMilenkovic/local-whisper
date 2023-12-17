import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size", type=str, required=True, help="Size of whisper model to train"
    )
    parser.add_argument(
        "--model_ckpt_location",
        type=str,
        required=True,
        help="Checkpoint of finetuned whisper model to evaluate",
    )

    args = parser.parse_args()
