import json

import os
from dotenv import load_dotenv
from threading import Thread

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse


from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    TextIteratorStreamer,
)


load_dotenv()

app = FastAPI()

WHISPER_CKPT_LOCATION = os.getenv("WHISPER_CKPT_LOCATION")

model = WhisperForConditionalGeneration.from_pretrained(WHISPER_CKPT_LOCATION)
processor = WhisperProcessor.from_pretrained(WHISPER_CKPT_LOCATION)
tokenizer = AutoTokenizer.from_pretrained(WHISPER_CKPT_LOCATION)


def get_transcription(audio_segment):
    waveform = audio_segment  # TODO: check what processing should be done here
    sampling_rate = 22050  # TODO: extract this from the audio segment
    streamer = TextIteratorStreamer(tokenizer)
    input_features = processor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    # streaming code based on https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L159
    generation_kwargs = dict(input_features, streamer=streamer, max_new_tokens=20)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(generated_text)
        yield new_text

    thread.join()


def convert_to_audio_segment(audio_file):
    # TODO: preprocessing...
    pass


@app.get("/transcribe")
def read_item(audio_file: UploadFile = File(...)):
    audio_segment = convert_to_audio_segment(audio_file.read())
    return StreamingResponse(get_transcription(audio_segment))
