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

INSTRUMENT = "instrumental.wav"
LYRICS = "lyrics.ass"
ORIGINAL = "original.wav"
VIDEO = "video_only.mp4"
VOCALS = "vocals.wav"
RAW = "raw.wav"
METADATA = "metadata.json"
OUTPUT = "output.mkv"

root_dir = Path("./data")

class ProcessorState(IntEnum):
    INIT = auto()
    METADATA_FETCHED = auto()
    DOWNLOADED = auto()
    SPLITTED = auto()
    DEMUCSD = auto()
    TRANSCRIBED = auto()
    BAKED = auto()

class YtSongProcessor:
    def __init__(self, yt_token):
        self.title = ""
        self.artist = ""
        self.device = "cuda" if os.environ.get("DEVICE") == "cuda" else "cpu"
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
                self.yt_token = yt_token
                self.dump()
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.state = ProcessorState.INIT
            self.yt_token = yt_token
            self.dump()
        
        print(f"[INFO] Processor initialized for {yt_token} at {self.path}, state: {self.state}")
    
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
                    self.demucs()
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
        if os.getenv("COOKIE_LOCATION") is not None:
            yt_dlp_cmd = [
                "yt-dlp",
                "--cookies", os.getenv("COOKIE_LOCATION") / "Cookies",
                "--force-overwrites",
                "-f", "mp4",
                "-o", str(path),
                youtube_url,
            ]
           
        if not run_command(yt_dlp_cmd):
            raise RuntimeError(f"yt-dlp failed for {self.yt_token}")
        self.state = ProcessorState.DOWNLOADED
        print(f"[INFO] Downloaded {self.yt_token} to {path}")
    
    def split(self):
        if self.state >= ProcessorState.SPLITTED:
            print(f"[INFO] Song {self.yt_token} already splitted, skip")
            return
        path = self.path / RAW
        if not path.exists():
            print(f"[ERROR] Song {self.yt_token} not downloaded, skip")
            return
        
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

        self.state = ProcessorState.SPLITTED
        print(f"[INFO] Splitted {self.yt_token} to {self.path}")
    
    def demucs(self):
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

        print(f"[INFO] Transcribing {self.yt_token} to {self.path}")
        step4_transcribe_stable_ts(vocals_path, ass_path, self.device)

        self.state = ProcessorState.TRANSCRIBED
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
        
        output_filename = f"{self.title}.mkv"
        output_path = self.path / output_filename
        if output_path.exists():
            print(f"[WARNING] Song {self.yt_token} already baked, probably dirty, cleaning...")
            os.remove(output_path)
        print(f"[INFO] Baking {self.yt_token} to {self.path}")
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(instrumental_path),
            "-i", str(vocals_path),
            "-i", str(ass_path),
            "-map", "0:v",
            "-map", "1:a",
            "-map", "2:a",
            "-map", "3:s",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k",
            "-c:s", "copy",
            "-metadata:s:a:0", "title=instrumental",
            "-metadata:s:a:1", "title=vocal",
            "-disposition:a:0", "default",
            str(output_path),
        ]
        if run_command(ffmpeg_cmd):
            self.state = ProcessorState.BAKED
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

def run_command(cmd, shell=False):
    """Utility to run external commands and log output."""
    print(f"[EXEC] Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, text=True, shell=shell, check=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {result.stderr}")
        return False
    return True

def step4_transcribe_stable_ts(audio_path, output_ass_path, device):
    print(f"[STEP 4] Transcribing using Stable-TS on {device}...")
    
    # 1. 加载模型
    model = stable_whisper.load_model("large-v3", device=device)
    
    # 2. 转录 (regroup=True 非常重要，它会把断开的词重新组装成句子)
    # result 是一个 stable_whisper.result.WhisperResult 对象
    result = model.transcribe(str(audio_path), language="zh", regroup=True)
    
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
        'extract_flat': False, # 确保获取完整层级信息
    }
    if os.getenv("COOKIE_LOCATION") is not None:
        ydl_opts['cookiefile'] = os.getenv("COOKIE_LOCATION") / "Cookies"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # download=False 仅提取元数据
            info = ydl.extract_info(url, download=False)
            
            # 1. 尝试获取 YouTube Music 的官方字段 (最精准)
            title = info.get('track') 
            artist = info.get('artist')

            # 2. 如果官方字段为空，尝试从视频标题中解析
            # 很多歌曲标题格式为 "Artist - Title" 或 "Title - Artist"
            if not title:
                full_title = info.get('title', 'Unknown_Title')
                if " - " in full_title:
                    parts = full_title.split(" - ", 1)
                    # 这里假设格式是 Artist - Title，你可以根据需要调换
                    artist = parts[0].strip()
                    title = parts[1].strip()
                else:
                    title = full_title
                    artist = info.get('uploader', 'Unknown_Artist')

            # 3. 数据清理：移除文件名非法字符（对你保存文件很有用）
            title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in ' .-_']).strip()
            artist = "".join([c for c in artist if c.isalpha() or c.isdigit() or c in ' .-_']).strip()

            return {
                "title": title,
                "artist": artist,
                "duration": info.get('duration'),
                "thumbnail": info.get('thumbnail'),
                "id": info.get('id')
            }
        except Exception as e:
            print(f"获取元数据失败: {e}")
            return None

if __name__ == "__main__":
    yt_token = "OGjXZro9-vM"
    processor = YtSongProcessor(yt_token)
    processor.run()
    