import json

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()


def fake_data_streamer(audio_segment):
    for i in range(10):
        yield i


@app.get("/transcribe")
def read_item(audio_file: UploadFile = File(...)):
    audio_segment = convert_to_audio_segment(await audio_file.read())
    return StreamingResponse(fake_data_streamer(audio_segment))
