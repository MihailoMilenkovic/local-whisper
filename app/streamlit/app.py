from typing import Optional
import os
import sys
import datetime
import argparse

import torch
import torchaudio
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import WhisperProcessor, WhisperForConditionalGeneration, PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_resample(
    audio,
    sample_rate: Optional[int] = None,
    required_sample_rate: Optional[int] = 16000,
):
    if isinstance(audio, torch.Tensor):
        waveform = audio
        assert sample_rate is not None
    else:
        waveform, sample_rate = torchaudio.load(audio)

    # Select the first channel if the audio has multiple channels
    # TODO: change to averaging samples across channels
    if waveform.shape[0] > 1:
        waveform = waveform[[0]]

    # Resample the audio if required
    if sample_rate != required_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=required_sample_rate
        )

    return waveform


def transcribe(
    audio,
    whisper_processor: WhisperProcessor,
    whisper_model: WhisperForConditionalGeneration,
):
    audio_tensor, audio_sampling_rate = load_and_resample(audio)
    input_features = whisper_processor(
        audio_tensor, sampling_rate=audio_sampling_rate, return_tensors="pt"
    ).input_features
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features.to(device))[0]
    transcription = whisper_processor.decode(predicted_ids)
    return transcription


def save_audio_file(audio_bytes, file_extension):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


def transcribe_audio(
    file_path: str, model: WhisperForConditionalGeneration, processor: WhisperProcessor
):
    """
    Transcribe the audio file at the specified path.

    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """
    with open(file_path, "rb") as audio_file:
        transcript = transcribe(
            audio_file, whisper_model=model, whisper_processor=processor
        )

    return transcript["text"]


def set_up_streamlit_app(
    model: WhisperForConditionalGeneration, processor: WhisperProcessor
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
            save_audio_file(audio_bytes, "mp3")

    # Upload Audio tab
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])
        if audio_file:
            file_extension = audio_file.type.split("/")[1]
            save_audio_file(audio_file.read(), file_extension)

    # Transcribe button action
    if st.button("Transcribe"):
        # Find the newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )

        # Transcribe the audio file
        transcript_text = transcribe_audio(
            audio_file_path, model=model, processor=processor
        )

        # Display the transcript
        st.header("Transcript")
        st.write(transcript_text)

        # Save the transcript to a text file
        with open("transcript.txt", "w") as f:
            f.write(transcript_text)

        # Provide a download button for the transcript
        st.download_button("Download Transcript", transcript_text)


def set_up_whisper(ckpt_location: str, lora_ckpt_location: Optional[str] = None):
    args = parser.parse_args()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_ckpt_location, device_map="auto"
    )
    if args.lora_ckpt_location:
        model = PeftModel.from_pretrained(model, args.lora_ckpt_location)

    processor = WhisperProcessor.from_pretrained(args.model_ckpt_location)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="sr", task="transcribe"
    )
    model = model.to(device)


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
    model, processor = set_up_whisper(
        model_ckpt=args.model_ckpt_location, lora_ckpt=args.lora_ckpt_location
    )

    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    set_up_streamlit_app()
