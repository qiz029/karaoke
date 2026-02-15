from fastapi import FastAPI
import uvicorn
from orchestration import YtSongProcessor
import asyncio
import modal_app
from typing import Annotated
from fastapi import FastAPI, Header

app = FastAPI()

@app.get("/")
def hello():
    return {"msg": "hello"}

@app.get(f"/song_process/{{yt_token}}")
def get_song_process_status(yt_token: str):
    processor = YtSongProcessor(yt_token)
    return {
        "yt_token": yt_token,
        "status": processor.state,
    }

@app.post(f"/song_process/{{yt_token}}/start")
async def start_song_process(yt_token: str, 
    device: Annotated[str | None, Header()] = None):
    if device is None:
        device = "cpu"
    print(f"[INFO] No device specified, using {device}")
    processor = YtSongProcessor(yt_token, device=device)
    asyncio.create_task(asyncio.to_thread(processor.run))
    return {
        "yt_token": yt_token,
        "status": processor.state,
    }

@app.post(f"/song_process/{{yt_token}}/clean")
def clean_song_process(yt_token: str):
    processor = YtSongProcessor(yt_token)
    processor.clean()
    return {
        "yt_token": yt_token,
        "status": processor.state,
    }

@app.post(f"/song_process/{{yt_token}}/step/{{step}}")
def start_song_process(
    yt_token: str,
    step: str,
    device: Annotated[str | None, Header()] = None,
):
    if device is None:
        device = "cpu"
    print(f"[INFO] No device specified, using {device}")
    processor = YtSongProcessor(yt_token, device=device)
    match step:
        case "download":
            processor.download()
        case "fetch_metadata":
            processor.fetch_metadata()
        case "demucs":
            processor.demucs()
        case "transcribe":
            processor.transcribe()
        case "split":
            processor.split()
        case "bake":
            processor.bake()
    return {
        "yt_token": yt_token,
        "status": processor.state,
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )