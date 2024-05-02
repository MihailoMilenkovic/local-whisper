from typing import Optional, Tuple
import os
import sys
import datetime
import argparse

import torch
import torchaudio
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)
from datasets import Audio, Dataset
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_SAMPLE_RATE = 16000


def transcribe(
    audio_file: str,
    whisper_model: WhisperForConditionalGeneration,
    feature_extractor: WhisperFeatureExtractor,
    processor: WhisperProcessor,
):
    audio_dataset = Dataset.from_dict({"audio": [audio_file]}).cast_column(
        "audio", Audio(sampling_rate=WHISPER_SAMPLE_RATE)
    )
    audio = audio_dataset[0]["audio"]
    print("AUDIO:", audio)
    input_features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features.to(device))[0]
    transcription = processor.decode(predicted_ids)
    normalized_transcription = processor.tokenizer._normalize(transcription)
    print("TRANSCRIPTION:", normalized_transcription)
    return normalized_transcription


def save_audio_file(audio_bytes, file_extension, audio_output_dir):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = os.path.join(audio_output_dir, f"audio_{timestamp}.{file_extension}")

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


def transcribe_audio(
    file_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    feature_extractor: WhisperFeatureExtractor,
):
    """
    Transcribe the audio file at the specified path.

    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """
    transcript = transcribe(
        audio_file=file_path,
        whisper_model=model,
        feature_extractor=feature_extractor,
        processor=processor,
    )

    return transcript


def set_up_streamlit_app(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    feature_extractor: WhisperFeatureExtractor,
    audio_output_dir: str,
):
    """
    Main function to run the Whisper Transcription app.
    """
    st.title("Whisper Transcription")

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    # Record Audio tab
    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes, "mp3", audio_output_dir=audio_output_dir)

    # Upload Audio tab
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])
        if audio_file:
            file_extension = audio_file.type.split("/")[1]
            save_audio_file(
                audio_file.read(), file_extension, audio_output_dir=audio_output_dir
            )

    # Transcribe button action
    if st.button("Transcribe"):
        # Find the newest audio file
        audio_file_path = max(
            [
                os.path.join(audio_output_dir, f)
                for f in os.listdir(audio_output_dir)
                if f.startswith("audio")
            ],
            key=os.path.getctime,
        )

        # Transcribe the audio file
        transcript_text = transcribe_audio(
            audio_file_path,
            model=model,
            processor=processor,
            feature_extractor=feature_extractor,
        )

        # Display the transcript
        st.header("Transcript")
        st.write(transcript_text)

        # Save the transcript to a text file
        with open("transcript.txt", "w") as f:
            f.write(transcript_text)

        # Provide a download button for the transcript
        st.download_button("Download Transcript", transcript_text)


def set_up_whisper(
    model_ckpt_location: str, lora_ckpt_location: Optional[str] = None
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor]:
    print("LOADING WHISPER MODEL FROM", model_ckpt_location)
    model = WhisperForConditionalGeneration.from_pretrained(model_ckpt_location)
    if lora_ckpt_location:
        print("LOADING WHISPER LORA CKPT FROM", lora_ckpt_location)
        model = PeftModel.from_pretrained(model, lora_ckpt_location)

    processor = WhisperProcessor.from_pretrained(model_ckpt_location)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_ckpt_location)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="sr", task="transcribe"
    )
    model = model.to(device)
    return model, processor, feature_extractor


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

    args = parser.parse_args()
    model, processor, feature_extractor = set_up_whisper(
        model_ckpt_location=args.model_ckpt_location,
        lora_ckpt_location=args.lora_ckpt_location,
    )
    working_dir = os.path.dirname(os.path.abspath(__file__))
    audio_output_dir = os.path.join(working_dir, "audio_files")
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    set_up_streamlit_app(
        model=model,
        processor=processor,
        feature_extractor=feature_extractor,
        audio_output_dir=audio_output_dir,
    )
