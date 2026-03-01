from fastapi import FastAPI
import uvicorn
from orchestration import YtSongProcessor
import asyncio
import modal_app
from typing import Annotated
from fastapi import FastAPI, Header
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pathlib import Path
import json
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

root_dir = Path("/Users/toddzheng/.ktv_control_plane/data")

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

@app.get("/list_songs")
async def list_processed_songs():
    processed_metadata = []
    
    # 确保根目录存在
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
        return {
            "song_list": []
        }

    # 遍历目录
    for p in root_dir.iterdir():
        if p.is_dir():
            metadata_path = p / "metadata.json"
            
            if metadata_path.exists():
                try:
                    # 使用 sync open 即可，JSON 读取通常很快
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 💡 关键改进：自动注入目录 ID 和音频路径
                        # 这样 Wails 前端拿到数据后才知道去哪里找音频文件
                        data["id"] = p.name  
                        # 假设你的音频文件叫 base.mp3 或 vocals.mp3
                        data["folder_path"] = str(p.absolute())
                        
                        processed_metadata.append(data)
                except (json.JSONDecodeError, OSError) as e:
                    # 在 FastAPI 中，生产环境建议使用 logger
                    print(f"Error parsing {metadata_path}: {e}")
            else:
                print(f"Skipping directory {p.name}: No metadata.json found")
                
    return {
        "song_list": processed_metadata
    }

@app.get("/download_mkv/{yt_token}")
async def download_mkv(yt_token: str):
    # 路径安全检查
    if ".." in yt_token or yt_token.startswith("/"):
        raise HTTPException(status_code=400, detail="非法路径")

    song_dir = root_dir / yt_token
    mkv_files = list(song_dir.glob("*.mkv"))
    
    if not mkv_files:
        raise HTTPException(status_code=404, detail="未找到 MKV 视频文件")
    
    # --- FIX: Grab the first item from the list ---
    file_path = mkv_files[0]
    
    return FileResponse(
        path=file_path,
        media_type='video/x-matroska',
        filename=file_path.name
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=14230,
        reload=True,
    )