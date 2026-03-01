from enum import auto
import subprocess
import shutil
import datetime
from pathlib import Path
from time import sleep
import stable_whisper
from enum import IntEnum
import json
import os
import yt_dlp
import syncedlyrics
import re
import modal_app

import modal

INSTRUMENT = "instrumental.wav"
LYRICS = "lyrics.ass"
ORIGINAL = "original.wav"
VIDEO = "video_only.mp4"
VOCALS = "vocals.wav"
RAW = "raw.wav"
METADATA = "metadata.json"
OUTPUT = "output.mkv"

root_dir = Path("/Users/toddzheng/.ktv_control_plane/data")
stable_ts_model = stable_whisper.load_model("large-v3", device="cpu")

class ProcessorState(IntEnum):
    INIT = auto()
    METADATA_FETCHED = auto()
    DOWNLOADED = auto()
    SPLITTED = auto()
    DEMUCSD = auto()
    TRANSCRIBED = auto()
    BAKED = auto()

process_state_m = {
    str(state): state.value for state in ProcessorState
}

class YtSongProcessor:
    def __init__(self, yt_token, device = "cpu"):
        self.title = ""
        self.artist = ""
        self.device = device
        self.state_timer = {}
        print(f"[INFO] Using device: {self.device}")
        if not yt_token:
            raise ValueError("yt_token is required")
        
        self.path = root_dir / yt_token
        if self.path.exists() and (self.path / METADATA).exists():
            try:
                self.load()
            except json.JSONDecodeError:
                print(f"[ERROR] Corrupted metadata file for {yt_token}, re-fetching...")
                os.remove(self.path / METADATA)
                self.state = ProcessorState.INIT
                self.state_timer[self.state] = datetime.datetime.now()
                self.yt_token = yt_token
                self.dump()
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.state = ProcessorState.INIT
            self.state_timer[self.state] = datetime.datetime.now()
            self.yt_token = yt_token
            self.dump()
        
        print(f"[INFO] Processor initialized for {yt_token} at {self.path}, state: {self.state}")
    
    def clean(self):
        if self.path and os.path.exists(self.path):
            try:
                # 2. 递归删除文件夹及其所有内容
                shutil.rmtree(self.path)
                print(f"✅ 已清空并删除目录: {self.path}")
                modal_app_clean(str(self.yt_token))
                print(f"✅ 已删除 Modal 数据卷: {self.yt_token}")
            except Exception as e:
                print(f"❌ 删除失败 {self.path}: {e}")

    def run(self):
        while self.state < ProcessorState.BAKED:
            match self.state:
                case ProcessorState.INIT:
                    self.fetch_metadata()
                case ProcessorState.METADATA_FETCHED:
                    self.download()
                case ProcessorState.DOWNLOADED:
                    self.split()
                case ProcessorState.SPLITTED:
                    self.demucs(self.device, version="v1")
                case ProcessorState.DEMUCSD:
                    self.transcribe()
                case ProcessorState.TRANSCRIBED:
                    self.bake()
                case ProcessorState.BAKED:
                    print(f"[INFO] Song {self.yt_token} already baked, finished.")
                    return
            self.dump()
        sleep(5)

    def fetch_metadata(self):
        if self.state >= ProcessorState.METADATA_FETCHED:
            print(f"[INFO] Song {self.yt_token} already fetched metadata, skip")
            return
        
        youtube_url = "https://www.youtube.com/watch?v=" + self.yt_token
        print(f"[INFO] Fetching metadata for {youtube_url} to {self.path / METADATA}")
        metadata_dict = get_video_metadata(youtube_url)
        self.title = metadata_dict["title"]
        self.artist = metadata_dict["artist"]
        self.state = ProcessorState.METADATA_FETCHED
        self.state_timer[self.state] = datetime.datetime.now()
        print(f"[INFO] Fetched metadata for {self.yt_token} to {self.path / METADATA}")

    def download(self):
        if self.state >= ProcessorState.DOWNLOADED:
            print(f"[INFO] Song {self.yt_token} already downloaded, skip")
            return
        path = self.path / RAW
        if path.exists():
            print(f"[WARNING] Song {self.yt_token} already downloaded, probably dirty, cleaning...")
            os.remove(path)
        
        youtube_url = "https://www.youtube.com/watch?v=" + self.yt_token
        print(f"[INFO] Downloading {youtube_url} to {path}")
        yt_dlp_cmd = [
            "yt-dlp",
            "--force-overwrites",
            "-f", "mp4",
            "-o", str(path),
            youtube_url,
        ]
           
        if not run_command(yt_dlp_cmd):
            raise RuntimeError(f"yt-dlp failed for {self.yt_token}")
        self.state = ProcessorState.DOWNLOADED
        self.state_timer[self.state] = datetime.datetime.now()
        print(f"[INFO] Downloaded {self.yt_token} to {path}")
    
    def split(self):
        if self.state >= ProcessorState.SPLITTED:
            print(f"[INFO] Song {self.yt_token} already splitted, skip")
            return
        path = self.path / RAW
        if not path.exists():
            print(f"[ERROR] Song {path} not downloaded, skip")
            raise FileNotFoundError(f"Song {self.yt_token} not downloaded, skip")
        
        original_path = self.path / ORIGINAL
        if original_path.exists():
            print(f"[WARNING] Song {self.yt_token} already splitted, probably dirty, cleaning...")
            os.remove(original_path)
        video_path = self.path / VIDEO
        if video_path.exists():
            print(f"[WARNING] Song {self.yt_token} already splitted, probably dirty, cleaning...")
            os.remove(video_path)

        print(f"[INFO] Splitting {self.yt_token} to {self.path}")

        if not run_command(["ffmpeg", "-y", "-i", str(path), "-an", "-vcodec", "copy", str(video_path)]):
            raise RuntimeError(f"FFmpeg failed for {self.yt_token}")
        if not run_command(["ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le", "-ar", "44100", str(original_path)]):
            raise RuntimeError(f"FFmpeg failed for {self.yt_token}")

        modal_app_upload_if_not_exists(str(original_path), self.yt_token + "/" + ORIGINAL)
        self.state = ProcessorState.SPLITTED
        self.state_timer[self.state] = datetime.datetime.now()
        print(f"[INFO] Splitted {self.yt_token} to {self.path}")
    
    def demucs(self, device = "cpu", version = "v1"):
        if self.state >= ProcessorState.DEMUCSD:
            print(f"[INFO] Song {self.yt_token} already demucs'd, skip")
            return
        original_path = self.path / ORIGINAL
        if not original_path.exists():
            print(f"[ERROR] Song {self.yt_token} not splitted, skip")
            return

        instrument_path = self.path / INSTRUMENT
        if instrument_path.exists():
            print(f"[WARNING] Song {self.yt_token} already demucs'd, probably dirty, cleaning...")
            os.remove(instrument_path)
        vocals_path = self.path / VOCALS
        if vocals_path.exists():
            print(f"[WARNING] Song {self.yt_token} already demucs'd, probably dirty, cleaning...")
            os.remove(vocals_path)

        if device == "cuda":
            print(f"[INFO] Demucs'ing {self.yt_token} to {self.path} on {device}")
            remote_instrumental = self.yt_token + "/" + INSTRUMENT
            remote_vocals = self.yt_token + "/" + VOCALS
            remote_original = self.yt_token + "/" + ORIGINAL
            instrument_exists = modal_app_check_file_exists(remote_instrumental)
            vocals_exists = modal_app_check_file_exists(remote_vocals)
            if not vocals_exists or not instrument_exists:
                if version == "v2":
                    print(f"[INFO] Demucs'ing {self.yt_token} to {self.path} on {device} with version {version}")
                    result = modal_app_demucs_v2(remote_original)
                elif version == "v3":
                    print(f"[INFO] Demucs'ing {self.yt_token} to {self.path} on {device} with version {version}")
                    result = modal_app_demucs_v3(remote_original)
                else:
                    print(f"[INFO] Demucs'ing {self.yt_token} to {self.path} on {device} with version {version}")
                    result = modal_app_demucs(remote_original)
                if not result:
                    raise RuntimeError(f"Demucs failed for {self.yt_token}")
            if not modal_app.download_if_not_exist(remote_instrumental, instrument_path):
                raise RuntimeError(f"Failed to download {remote_instrumental} to {instrument_path}")
            if not modal_app.download_if_not_exist(remote_vocals, vocals_path):
                raise RuntimeError(f"Failed to download {remote_vocals} to {vocals_path}")
        else:
            tmp_dir = self.path / "tmp"
            if tmp_dir.exists():
                print(f"[WARNING] Song {self.yt_token} already demucs'd, probably dirty, cleaning...")
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[INFO] Demucs'ing {self.yt_token} to {self.path}")
            demucs_cmd = ["demucs", "--two-stems", "vocals", "-o", str(tmp_dir), "--device", self.device, str(original_path)]
            if run_command(demucs_cmd):
                # Move files to work_dir root
                sep_path = tmp_dir / "htdemucs" / "original"
                shutil.move(str(sep_path / "vocals.wav"), str(vocals_path))
                shutil.move(str(sep_path / "no_vocals.wav"), str(instrument_path))
                shutil.rmtree(tmp_dir)
            else:
                raise RuntimeError(f"Demucs failed for {self.yt_token}")
        
        self.state = ProcessorState.DEMUCSD
        self.state_timer[self.state] = datetime.datetime.now()
        print(f"[INFO] Demucs'd {self.yt_token} to {self.path}")

    def transcribe(self):
        if self.state >= ProcessorState.TRANSCRIBED:
            print(f"[INFO] Song {self.yt_token} already transcribed, skip")
            return
        vocals_path = self.path / VOCALS
        if not vocals_path.exists():
            print(f"[ERROR] Song {self.yt_token} not demucs'd, skip")
            return

        ass_path = self.path / LYRICS
        if ass_path.exists():
            print(f"[WARNING] Song {self.yt_token} already transcribed, probably dirty, cleaning...")
            os.remove(ass_path)

        print(f"[INFO] Transcribing {self.yt_token} to {self.path} using {self.device}")
        if self.device == "cuda":
            remote_audio = self.yt_token + "/" + VOCALS
            remote_ass = self.yt_token + "/" + LYRICS
            if not modal_app_check_file_exists(remote_ass):
                modal_app_transcribe(self.title, self.artist, remote_audio)
            if not modal_app.download_if_not_exist(remote_ass, ass_path):
                raise RuntimeError(f"Failed to download {remote_ass} to {ass_path}")
        else:
            step4_transcribe_stable_ts(self.title, vocals_path, ass_path, "cpu")
        
        self.state = ProcessorState.TRANSCRIBED
        self.state_timer[self.state] = datetime.datetime.now()
        print(f"[INFO] Transcribed {self.yt_token} to {self.path}")

    def bake(self):
        if self.state >= ProcessorState.BAKED:
            print(f"[INFO] Song {self.yt_token} already baked, skip")
            return
        video_path = self.path / VIDEO
        if not video_path.exists():
            print(f"[ERROR] Song {self.yt_token} not splitted, skip")
            return
        instrumental_path = self.path / INSTRUMENT
        if not instrumental_path.exists():
            print(f"[ERROR] Song {self.yt_token} not demucs'd, skip")
            return
        vocals_path = self.path / VOCALS
        if not vocals_path.exists():
            print(f"[ERROR] Song {self.yt_token} not demucs'd, skip")
            return
        ass_path = self.path / LYRICS
        if not ass_path.exists():
            print(f"[ERROR] Song {self.yt_token} not transcribed, skip")
            return
        original_path = self.path / ORIGINAL
        if not original_path.exists():
            print(f"[ERROR] Song {self.yt_token} not splitted, skip")
            return
        
        output_filename = f"{self.title}.mkv"
        output_path = self.path / sanitize_filename(output_filename)
        if output_path.exists():
            print(f"[WARNING] Song {self.yt_token} already baked, probably dirty, cleaning...")
            os.remove(output_path)
        print(f"[INFO] Baking {self.yt_token} to {self.path}")
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", str(instrumental_path),
            "-i", str(vocals_path),
            "-i", str(ass_path),
            "-i", str(original_path),
            "-map", "0:v",
            "-map", "1:a",
            "-map", "2:a",
            "-map", "3",
            "-map", "4:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k",
            "-c:s", "copy",
            "-metadata:s:a:0", "title=instrumental",
            "-metadata:s:a:1", "title=vocal",
            "-metadata:s:a:2", "title=original",
            "-disposition:a:0", "default",
            "-disposition:s:0", "default",
            str(output_path),
        ]
        if run_command(ffmpeg_cmd):
            self.state = ProcessorState.BAKED
            self.state_timer[self.state] = datetime.datetime.now()
            print(f"[INFO] Baked {self.yt_token} to {self.path}")
        else:
            raise RuntimeError(f"FFmpeg failed for {self.yt_token}")


    def dump(self):
        with open(self.path / METADATA, "w", encoding='utf-8') as f:
            dict = {
                "yt_token": self.yt_token,
                "state": self.state.value,
                "path": str(self.path),
                "title": self.title,
                "artist": self.artist,
                "state_timer": {str(k): v.isoformat() for k, v in self.state_timer.items()},
            }
            json.dump(dict, f, indent=4, ensure_ascii=False)

    def load(self):
        metadata_path = self.path / METADATA
        if not metadata_path.exists():
            self.dump()
        with open(self.path / METADATA, "r", encoding='utf-8') as f:
            dict = json.load(f)
            self.yt_token = dict["yt_token"]
            self.state = ProcessorState(dict["state"])
            self.path = Path(dict["path"])
            self.title = dict["title"]
            self.artist = dict["artist"]
            if "state_timer" in dict:
                self.state_timer = {process_state_m[k]: datetime.datetime.fromisoformat(v) for k, v in dict["state_timer"].items()}
            else:
                self.state_timer = {}

def run_command(cmd, shell=False):
    """Utility to run external commands and log output."""
    print(f"[EXEC] Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, text=True, shell=shell, check=True, capture_output=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {result.stderr}")
        return False
    return True

def get_plain_lyrics(song_name):
    """
    1. 搜索歌词 (返回的是 LRC 格式)
    2. 去除 LRC 里的时间戳，只保留纯文本
    """
    print(f"正在搜索歌词: {song_name} ...")
    
    # providers=["netease"] 对中文歌最友好，也可以加上 "musixmatch"
    lrc_content = syncedlyrics.search(song_name, providers=["netease"])
    
    if not lrc_content:
        print("未找到歌词，将回退到纯 AI 转录模式。")
        return None

    # 清洗数据：去除 [00:12.34] 这种标签，只留歌词文本
    plain_lines = []
    for line in lrc_content.split('\n'):
        # 正则替换掉时间轴
        clean_line = re.sub(r'\[.*?\]', '', line).strip()
        if clean_line:
            plain_lines.append(clean_line)
            
    # 合并成一段长文本，中间用换行符或者空格隔开都可以
    # stable-ts 会自动处理
    return "\n".join(plain_lines)

def step4_transcribe_stable_ts(song_name, audio_path, output_ass_path, device):
    print(f"[STEP 4] Transcribing using Stable-TS on {device}...")

    # 2. 获取正确文本
    correct_text = get_plain_lyrics(song_name)
    
    if correct_text:
        print("✅ 找到官方歌词，正在进行强制对齐 (Alignment)...")
        # 3a. 核心步骤：align()
        # 这里不需要 transcribe，直接告诉 AI：“就是这些字，你帮我把时间对上”
        result = stable_ts_model.align(
            str(audio_path), 
            correct_text, 
            language="zh", 
            original_split=True # 尝试保留原歌词的换行结构
        )
    else:
        print("未找到官方歌词，回退到普通转录模式...")
        # 3b. 普通转录模式
        result = stable_ts_model.transcribe(str(audio_path), language="zh", regroup=True)
    
    # 3. 关键转换！
    # 我们不调用 result.to_ass()，因为它的样式太丑且逻辑不符合 KTV。
    # 我们把 result 转成 dict，直接喂给我写的那个 generate_karaoke_ass 函数。
    # stable-ts 的 result.to_dict() 结构和 WhisperX/OpenAI 是一模一样的。
    data = result.to_dict()

    tmp_whisper_file = output_ass_path.parent / "tmp_whisper.json"
    with open(tmp_whisper_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # 4. 生成专业 KTV 字幕
    # 这里调用的是你脚本里定义的那个函数
    generate_karaoke_ass(data, output_ass_path)
    
    print(f"[INFO] Saved Customized KTV ASS to {output_ass_path}")

def sanitize_filename(name: str) -> str:
    # 替换掉路径分隔符 /
    # 也可以顺手把 : ? " < > | * 这些在 Windows 上非法但在 Mac 上合法的字符也处理掉，防患未然
    return name.replace("/", "_").replace("|", "-").replace(":", "-")

def format_ass_timestamp(seconds):
    """Converts seconds to ASS timestamp format H:MM:SS.cs"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    centiseconds = int(round((seconds - total_seconds) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def generate_karaoke_ass(whisper_result, output_path, max_chars_per_line=16):
    """
    生成 KTV 字幕：长句自动折行 (Wrap)，而不是切分。
    max_chars_per_line: 超过这个字数，就在中间插入换行符
    """
    
    # === 1. 定义 ASS 头部 ===
    # 注意 WrapStyle: 2 (不自动换行)，我们要自己控制换行
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "WrapStyle: 2\n" 
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Alignment: 2 (底部居中)。这样换行时，第一行会在第二行的上面。
        "Style: KTV,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,135,135,60,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    lines = [header]
    
    if "segments" not in whisper_result:
        print("[WARN] No segments found.")
        return

    # === 2. 主循环处理 ===
    for segment in whisper_result["segments"]:
        start_t = format_ass_timestamp(segment["start"])
        end_t = format_ass_timestamp(segment["end"])
        
        words = segment.get("words", [])
        text_len = len(segment.get("text", ""))
        
        # --- 核心逻辑：计算换行点 ---
        split_idx = -1
        # 如果句子总长度超过限制 (比如 16个字)
        if text_len > max_chars_per_line and len(words) > 1:
            # 我们在单词列表的中间位置插入换行符
            split_idx = len(words) // 2 

        ktv_text = ""
        cursor = segment["start"]

        if words:
            for i, word in enumerate(words):
                # 1. 检查是否需要插入换行符
                # 如果当前词是前半段的最后一个词之后，插入 \N
                if i == split_idx:
                    ktv_text += r"\N" # ASS 的硬换行符
                
                # 2. 处理时间轴 (Gap & Duration)
                w_start = word.get("start", cursor)
                w_end = word.get("end", w_start + 0.1)
                
                # Gap (空隙)
                gap = int(round((w_start - cursor) * 100))
                if gap > 1:
                    ktv_text += f"{{\\k{gap}}}"
                
                # Fill (歌词变色)
                duration = int(round((w_end - w_start) * 100))
                word_text = word["word"]
                
                ktv_text += f"{{\\kf{duration}}}{word_text}"
                
                cursor = w_end
        else:
            # Fallback: 没有词级时间轴时的处理
            raw_text = segment["text"]
            # 如果没时间轴但也太长，手动强行插个 \N 在中间
            if len(raw_text) > max_chars_per_line:
                mid = len(raw_text) // 2
                ktv_text = raw_text[:mid] + r"\N" + raw_text[mid:]
            else:
                ktv_text = raw_text

        lines.append(f"Dialogue: 0,{start_t},{end_t},KTV,,0,0,0,,{ktv_text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Wrapped ASS file saved to: {output_path}")

def get_video_metadata(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            print(info.keys())
            
            # --- 修复 2：确保变量都有默认值 ---
            # 优先获取官方元数据，默认为 None
            title = info.get('track') 
            if not title:
                title = info.get('title')
            artist = info.get('artist')

            # 如果没有官方标题，则尝试解析视频标题
            if not title:
                full_title = info.get('title', 'Unknown_Title')
                
                # 常见格式解析：Artist - Title 或 Title - Artist
                # 这里是个难点，通常欧美是 Artist - Title，但最好有个兜底
                if " - " in full_title:
                    parts = full_title.split(" - ", 1)
                    # 假设格式: Artist - Title
                    temp_artist = parts[0].strip()
                    temp_title = parts[1].strip()
                    
                    title = temp_title
                    # 只有当官方 artist 也没拿到时，才用解析出来的 artist
                    if not artist:
                        artist = temp_artist
                else:
                    title = full_title
            
            # 兜底：如果折腾半天 artist 还是空的，就用上传者名字
            if not artist:
                artist = info.get('uploader', 'Unknown_Artist')

            return {
                "title": title,
                "artist": artist,
            }
        except Exception as e:
            print(f"获取元数据失败: {e}")
            return None

def modal_app_check_file_exists(remote_path: str) -> bool:
    fn = modal.Function.from_name("ktv-processor", "check_file_exists")
    result_tuple = fn.remote(remote_path)
    
    exists = result_tuple[0]  # 或者写成: exists, _ = fn.remote(remote_path)
    
    return exists

def modal_app_demucs(remote_path: str) -> bool:
    fn = modal.Function.from_name("ktv-processor", "demucsFn")
    return fn.remote(remote_path)

def modal_app_demucs_v2(remote_path: str) -> bool:
    fn = modal.Function.from_name("ktv-processor", "demucs_fn_v2")
    return fn.remote(remote_path)

def modal_app_demucs_v3(remote_path: str) -> bool:
    fn = modal.Function.from_name("ktv-processor", "process_audio_for_ktv")
    return fn.remote(remote_path)

def modal_app_transcribe(title: str, artist: str, remote_path: str) -> bool:
    fn = modal.Function.from_name("ktv-processor", "transcribe_direct")
    return fn.remote(title, artist, remote_path)

def modal_app_clean(remote_path: str) -> bool:
    cmd = [
        "modal", "volume", "rm", "data", "-r", remote_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def modal_app_upload_if_not_exists(local_path: str, remote_filename: str):
    local_file = Path(local_path)
    if not local_file.exists():
        raise FileNotFoundError(f"本地文件未找到: {local_path}")

    print(f"🔍 检查云端文件: {remote_filename} ...")
    
    # 调用上面的云端函数进行检查
    exists = modal_app_check_file_exists(remote_filename)
    
    if exists:
        print(f"✅ 文件已存在: {remote_filename}，跳过上传。")
        return

    print(f"📤 文件不存在，开始上传: {local_path} (这可能需要一点时间)...")
    
    # 【关键修改】使用 Modal CLI 上传
    # 这绕过了 Python 函数参数的大小限制，支持断点续传和大文件
    # 格式: modal volume put <Volume名> <本地路径> <远程目标路径>
    cmd = [
        "modal", "volume", "put", 
        "data",           # Volume 名称
        str(local_file),      # 本地文件
        remote_filename       # 远程路径 (注意：这里不需要加 /data 前缀，CLI会自动处理)
    ]
    
    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"🎉 上传成功: {remote_filename}")
    else:
        # 如果出错，打印错误日志
        raise RuntimeError(f"❌ 上传失败: {result.stderr}")
