from time import sleep
import modal
import modal
import os
import subprocess
from pathlib import Path
import shutil
from pathlib import Path
import json

import datetime
import subprocess
import shutil
import datetime
from pathlib import Path
import json
import os

app = modal.App("ktv-processor")

def correct_lyrics_with_gemini(api_key: str, raw_text: str, song_metadata: str) -> str:
    from google import genai
    from google.genai import types
    """
    使用 Gemini 搜索并修正歌词
    raw_text: Whisper 识别出的粗糙文本
    song_metadata: "歌名 - 歌手"
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    你是一个专业的字幕校对员，充分利用你的搜索功能，找到这首歌曲的官方正确歌词。
    
    任务：
    1. 我会给你一段由 AI 语音识别生成的粗糙歌词（可能包含错别字、同音词错误）。
    2. 歌曲信息是：【{song_metadata}】。
    3. 请利用你的搜索能力，找到这首歌的**官方正确歌词**。
    4. 对比我提供的粗糙文本，输出修正后的、分行正确的**纯歌词**。
    5. **严禁**输出任何时间轴、解释、前言或后缀。只输出歌词内容。
    6. 保持原曲的段落结构。

    严格指令：
    绝对不要增减任何一行！保持原有的行数和结构完全一致。

    绝对不要删除重复的句子！歌手唱了几遍就保留几遍。

    绝对不要添加任何诸如 [副歌]、[间奏] 之类的标签。

    你的唯一任务是：只修改同音错别字，让句子通顺。其他一概不准动！"
    
    粗糙歌词输入：
    {raw_text}
    """
    tools = [
        types.Tool(
            google_search=types.GoogleSearch() # 启用内置搜索
        )
    ]
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview", # 推荐用 Flash，速度快且搜索能力强
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=tools,
                response_modalities=["TEXT"], # 确保只返回文本
                temperature=0.1 # 低温度，确保准确性
            )
        )
        
        # 3. 获取结果
        # 新 SDK 的 response.text 直接可用，如果有 grounding metadata 也可以在这里查
        corrected_text = response.text.strip()
        
        # 简单的后处理：去除可能的 Markdown 代码块标记
        corrected_text = corrected_text.replace("```lyric", "").replace("```text", "").replace("```", "").strip()
        return corrected_text
    except Exception as e:
        print(f"Gemini 修正失败: {e}")
        return raw_text # 失败降级：直接返回原文本

# 1. 定义一个持久化存储卷，名字叫 "ktv-data"
# create_if_missing=True 会自动创建它
volume = modal.Volume.from_name("data", create_if_missing=True)

# 2. 定义一个“搬运工”函数
# 它挂载了 volume 到 /data 目录
@app.function(volumes={"/data": volume})
def save_file_to_volume(file_content: bytes, remote_filename: str):
    remote_path = f"/data/{remote_filename}"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(remote_path), exist_ok=True)
    
    print(f"正在写入文件到: {remote_path}...")
    with open(remote_path, "wb") as f:
        f.write(file_content)
    
    # 【关键】强制提交更改，这样其他函数（比如你的 GPU 函数）能立即看到这个文件
    volume.commit() 
    print("写入完成并已提交 Volume。")

# ---------------------------------------------------------
# 1. 定义一个云端助手函数：只负责检查文件是否存在
# ---------------------------------------------------------
@app.function(volumes={"/data": volume})
def check_file_exists(remote_filename: str):
    """
    运行在云端，检查文件是否已经在 Volume 里了
    """
    remote_path = Path("/data") / remote_filename
    exists = remote_path.exists()
    if not exists:
        return exists, 0
    size_mb = remote_path.stat().st_size / (1024 * 1024) if exists else 0
    return exists, size_mb

image = (
    modal.Image.debian_slim()
    .pip_install("demucs", "torch", "torchaudio", "soundfile", "numpy", "stable-ts", "faster-whisper")
    # 2. 【关键】安装系统级依赖 ffmpeg (包含 ffprobe)
    .apt_install("ffmpeg", "libsndfile1")
    # 3. 【关键修改】把 google-genai 单独放一行，强制重新构建
    # 同时指定版本号 (>=0.3.0) 确保是支持 from google import genai 的新版
    .pip_install("google-genai>=0.3.0")
    .pip_install("audio-separator[gpu]")
    # 预下载模型到镜像里，加快启动速度
    .run_commands("python3 -c 'from demucs.pretrained import get_model; get_model(\"htdemucs\")'")
)

@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    timeout=600
)
def demucsFn(remote_filename: str):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model  # <--- 修改这里
    from demucs.audio import AudioFile, save_audio
    import soundfile as sf  # <--- 引入 soundfile

    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.audio import save_audio, AudioFile

    print(f"🚀 [Remote] 开始处理: {remote_filename}")
    
    input_path = Path("/data") / remote_filename
    output_dir = input_path.parent
    
    # 加载模型
    model = get_model("htdemucs")
    model.to("cuda")
    
    # 读取音频
    wav = AudioFile(input_path).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels
    )
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    
    # <--- 修改这里: 使用 apply_model
    sources = apply_model(
        model, 
        wav[None], 
        device="cuda", 
        shifts=1, 
        split=True, 
        overlap=0.25, 
        progress=True
    )[0]
    
    sources = sources * ref.std() + ref.mean()
    
    # 后续处理保持不变...
    vocab_idx = model.sources.index("vocals")
    vocals_wav = sources[vocab_idx]
    
    other_indices = [i for i in range(len(model.sources)) if i != vocab_idx]
    instr_wav = torch.zeros_like(vocals_wav)
    for i in other_indices:
        instr_wav += sources[i]
        
    vocals_out = output_dir / "vocals.wav"
    instr_out = output_dir / "instrumental.wav"
    
    print(f"💾 保存人声到: {vocals_out}")
    
    # 【关键修改】使用 soundfile 直接写入
    # 1. .cpu().numpy(): 把 Tensor 转成 numpy 数组
    # 2. .T: 转置。因为 Tensor 是 [声道, 时长]，但 soundfile 需要 [时长, 声道]
    sf.write(str(vocals_out), vocals_wav.cpu().numpy().T, model.samplerate)
    
    print(f"💾 保存伴奏到: {instr_out}")
    sf.write(str(instr_out), instr_wav.cpu().numpy().T, model.samplerate)
    
    return {
        "vocals": str(vocals_out),
        "instrumental": str(instr_out)
    }

@app.function(image=image, gpu="L4", volumes={"/data": volume})
def demucs_fn_v2(remote_path: str) -> bool:
    from audio_separator.separator import Separator
    import shutil
    
    base_path = Path("/data") / remote_path
    base_path = base_path.parent
    input_file = base_path / "original.wav"
    model_path = Path("model")
    
    # 初始化
    separator = Separator(
        model_file_dir=model_path,  # <--- 【关键】告诉库：去这个目录找文件
        output_dir=str(base_path),
        output_format="wav"
    )

    # --- 第一步：用 ViperX 做高质量基底分离 ---
    print("STEP 1: 分离伴奏与人声 (ViperX)...")
    separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    outputs_1 = separator.separate(str(input_file))
    
    # 假设 outputs_1[0] 是 Instrumental, outputs_1[1] 是 Vocals
    # 注意：audio-separator 的返回顺序可能需要通过文件名判断，这里简化处理
    inst_path = base_path / outputs_1[0] 
    vocals_path = base_path / outputs_1[1]

    # --- 第二步：从人声中提取和声 (Karaoke Model) ---
    print("STEP 2: 从人声中提取和声 (Mel-Band Karaoke)...")
    # 这个模型专门把 Vocals 拆成 "Lead" 和 "Backing"
    separator.load_model(model_filename="mel_band_roformer_karaoke_becruily.ckpt")
    outputs_2 = separator.separate(str(vocals_path))
    
    # outputs_2 里应该有一个是 backing vocals (和声)
    # 假设 outputs_2[0] 是 backing, outputs_2[1] 是 lead
    backing_path = base_path / outputs_2[0]
    lead_vocal_path = base_path / outputs_2[1]
    os.rename(lead_vocal_path, base_path / "vocals.wav")
    
    # --- 第三步：合并 (ffmpeg) ---
    print("STEP 3: 合并 纯伴奏 + 和声...")
    # 使用 ffmpeg mix 两个音频
    final_inst = base_path / "instrumental.wav"
    
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", str(inst_path),
        "-i", str(backing_path),
        "-filter_complex", "amix=inputs=2:duration=longest",
        str(final_inst)
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 合并失败: {e}")
        return False
    
    print(f"🎉 完美 KTV 伴奏已生成: {final_inst}")
    return True

@app.function(
    image=image,
    gpu="L4",  # L4 跑这个模型非常快，性价比最高
    volumes={"/data": volume},
    timeout=600
)
def process_audio_for_ktv(remote_filename: str):
    from audio_separator.separator import Separator

    print(f"🚀 [Remote] 开始高质量 KTV 分离: {remote_filename}")
    
    # 路径处理 (保持和你原逻辑一致)
    input_path = Path("/data") / remote_filename
    output_dir = input_path.parent
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化分离器
    # model_file_dir='/data/models': 把模型缓存在 Volume 里，防止每次冷启动都下载
    separator = Separator(
        log_level='INFO',
        model_file_dir='/data/models', 
        output_dir=str(output_dir),
        output_format='wav',
        normalization_threshold=0.9 # 防止爆音
    )

    # 加载 KTV 专用模型 (保留和声)
    # 第一次运行会自动下载
    separator.load_model(model_filename='UVR-MDX-NET-Karaoke-2.mdx')

    # 执行分离
    print(f"🔄 正在运行 UVR-MDX-NET-Karaoke-2 推理...")
    output_files = separator.separate(str(input_path))
    
    # --- 关键步骤：重命名以匹配接口 ---
    # audio-separator 生成的文件名通常带有模型后缀，比如 "原文件名_(Vocals)_UVR...wav"
    # 我们需要把它们重命名为 "vocals.wav" 和 "instrumental.wav"
    
    final_vocals_path = output_dir / "vocals.wav"
    final_instr_path = output_dir / "instrumental.wav"

    for fname in output_files:
        original_file_path = output_dir / fname
        
        # 逻辑判断：哪个是人声，哪个是伴奏
        # UVR-MDX-NET-Karaoke-2 的输出通常包含 "Vocals" 和 "Instrumental"
        if "Vocals" in fname:
            # 移动并覆盖 (如果有旧文件)
            shutil.move(str(original_file_path), str(final_vocals_path))
            print(f"💾 重命名人声为: {final_vocals_path}")
            
        elif "Instrumental" in fname:
            shutil.move(str(original_file_path), str(final_instr_path))
            print(f"💾 重命名伴奏为: {final_instr_path}")

    # 清理显存
    del separator
    import torch
    torch.cuda.empty_cache()

    # 返回和你原函数完全一致的字典结构
    return {
        "vocals": str(final_vocals_path),
        "instrumental": str(final_instr_path)
    }

@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("app-secret")],
    timeout=600
)
def transcribe(title: str, artist: str, audio_path: str):
    import stable_whisper
    import os
    import json
    from pathlib import Path
    
    device = 'cuda'
    print(f"[STEP 1] Transcribing using Stable-TS on {device}...")
    input_path = Path("/data") / audio_path
    output_dir = input_path.parent
    output_ass_path = output_dir / "lyrics.ass"

    # L4 显卡跑 float16 非常快
    model = stable_whisper.load_faster_whisper(
        "large-v3-turbo", 
        device="cuda",
        compute_type="float16" 
    )

    # 1. 初始转录（提取时间轴和初始文本）
    result = model.transcribe(
        str(input_path),
        language="zh",
        regroup=True,
        vad=True,          
        vad_parameters={
            "threshold": 0.5,               
            "min_speech_duration_ms": 300,  
            "min_silence_duration_ms": 500, 
        },
    )
    
    # 2. 调用大模型修正文本
    gemini_key = os.environ["GEMINI_KEY"]
    text = correct_lyrics_with_gemini(gemini_key, result.text, f"{title} - {artist}")

    print("📝 修正后的歌词预览:")
    print(text[:100] + "...")

    # === 3. Align (将修正后的文本强制对齐到音频) ===
    print("[STEP 3] Aligning text to audio...")
    result = model.align(
        str(input_path), 
        text, 
        language="zh",
        interpolate=True,
        original_split=True  # 保留 Gemini 的分行
    )
    
    # === 4. 关键步骤: Refine (修复赋值 Bug) ===
    print("[STEP 4] Refining timestamps...")
    # 【修复】必须把返回值重新赋给 result
    result = model.refine(
        str(input_path),
        result,  
        precision=0.05  # 精度设为 50ms，对 KTV 字幕来说足够平滑且准确
    )
    result.regroup(regroup_strategy='ms_we_sp', max_chars=20)
    
    # 5. 导出并生成专业 KTV 字幕
    data = result.to_dict()

    tmp_whisper_file = output_ass_path.parent / "tmp_whisper.json"
    with open(tmp_whisper_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # 调用你的字幕生成函数
    generate_karaoke_ass_v2(data, output_ass_path)
    
    print(f"[INFO] Saved Customized KTV ASS to {output_ass_path}")

@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("app-secret")],
    timeout=600
)
def transcribe_direct(title: str, artist: str, audio_path: str):
    import stable_whisper
    import json
    from pathlib import Path
    
    device = 'cuda'
    print(f"[STEP 1] Loading Stable-TS on {device}...")
    input_path = Path("/data") / audio_path
    output_dir = input_path.parent
    output_ass_path = output_dir / "lyrics.ass"

    # L4 显卡跑 float16 非常快
    model = stable_whisper.load_faster_whisper(
        "large-v3-turbo", 
        device="cuda",
        compute_type="float16" 
    )

    # === 1. 直接提取带字级时间轴的原始文本 ===
    print("[STEP 2] Transcribing and generating native word-level timestamps...")
    result = model.transcribe(
        str(input_path),
        language="zh",
        initial_prompt="这是一段纯净的歌词录音，不包含任何字幕组信息或片头广告。",
        regroup=True,
        word_timestamps=True,  # 【关键】确保强行提取字级时间戳
        vad=True,          
        vad_parameters={
            "threshold": 0.5,               
            "min_speech_duration_ms": 300,  
            "min_silence_duration_ms": 500, 
        },
    )
    
    # ⚠️ 【已经删除】Gemini 修正文本的步骤
    # ⚠️ 【已经删除】model.align 重新对齐的步骤

    # === 2. 关键步骤: Refine (巩固原始时间轴) ===
    print("[STEP 3] Refining timestamps...")
    result = model.refine(
        str(input_path),
        result,          # 直接把 transcribe 生成的结构完美的 result 传进来
        precision=0.05   # 50ms 精度
    )
    result.regroup(
        regroup_strategy='ms_we_sp', 
        max_chars=20,       # 根据你的视频宽度调整，一般 20 个汉字左右比较合适
        max_gap=1.5         # 唱词停顿超过 1.5 秒就换行，避免字幕在屏幕上停留太久
    )
    
    # === 3. 导出并生成专业 KTV 字幕 ===
    data = result.to_dict()

    tmp_whisper_file = output_ass_path.parent / "tmp_whisper.json"
    with open(tmp_whisper_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # 调用你的字幕生成函数
    generate_karaoke_ass_v2(data, output_ass_path)
    
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

def generate_karaoke_ass_v2(whisper_result, output_path, max_chars_per_line=16):
    """
    生成 KTV 字幕：
    1. 强制以 word['start'] 为基准，忽略 segment 造成的漂移。
    2. 增加 Lead-in (提前进场时间)，避免歌词跳出来瞬间就开始唱。
    """
    
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
        "Style: KTV,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,135,135,60,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    lines = [header]
    
    if "segments" not in whisper_result:
        return

    # 预留时间：字幕比人声早出来多少秒 (通常 0.5s - 1.0s)
    LEAD_IN_BUFFER = 0.5 
    # 退出缓冲：字幕唱完后多停留多久
    LEAD_OUT_BUFFER = 0.5

    for segment in whisper_result["segments"]:
        words = segment.get("words", [])
        
        # === 核心修正 1: 兜底处理 ===
        if not words:
            # 如果没有词级时间轴（对齐失败），回退到 segment 时间
            start_t = format_ass_timestamp(segment["start"])
            end_t = format_ass_timestamp(segment["end"])
            lines.append(f"Dialogue: 0,{start_t},{end_t},KTV,,0,0,0,,{segment['text']}")
            continue

        # === 核心修正 2: 重构 Line Start/End ===
        # 即使 segment['start'] 是 10.0s (因为包含了被删的语气词)，
        # 如果 words[0]['start'] 是 15.0s，我们就从 14.5s 开始显示字幕。
        
        real_start = words[0]['start']
        real_end = words[-1]['end']
        
        # 计算 ASS 行的显示时间
        ass_start_time = max(0, real_start - LEAD_IN_BUFFER)
        ass_end_time = real_end + LEAD_OUT_BUFFER
        
        start_t = format_ass_timestamp(ass_start_time)
        end_t = format_ass_timestamp(ass_end_time)

        # === 核心修正 3: 计算折行 ===
        text_len = len(segment.get("text", ""))
        split_idx = -1
        if text_len > max_chars_per_line and len(words) > 1:
            split_idx = len(words) // 2 

        ktv_text = ""
        
        # cursor 现在的初始值是 ASS 行的开始时间 (比如 14.5s)
        # 而第一个词开始是 15.0s
        # 它们之间的差值 (0.5s) 就是第一个 \k 的等待时间
        cursor = ass_start_time

        for i, word in enumerate(words):
            if i == split_idx:
                ktv_text += r"\N"

            w_start = word.get("start")
            w_end = word.get("end")

            # --- 关键 Gap 计算 ---
            # 计算当前词开始时间 与 光标位置 的差值
            # 只有当 w_start > cursor 时才会有正值的 gap
            # round(x * 100) 转换成厘秒
            gap_duration = int(round((w_start - cursor) * 100))
            
            # 如果 Gap 只有 1cs (0.01s) 这种极小值，往往是浮点误差，可以忽略
            # 但如果是首词 (i==0)，gap_duration 包含了 LEAD_IN，必须保留
            if gap_duration > 0:
                ktv_text += f"{{\\k{gap_duration}}}"
            
            # --- 单词持续时间 ---
            word_duration = int(round((w_end - w_start) * 100))
            # 最小给 1cs，防止 \kf0 导致渲染器报错
            word_duration = max(1, word_duration)
            
            ktv_text += f"{{\\kf{word_duration}}}{word['word']}"
            
            # 更新光标到当前词结束
            cursor = w_end

        lines.append(f"Dialogue: 0,{start_t},{end_t},KTV,,0,0,0,,{ktv_text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Robust KTV ASS file saved to: {output_path}")

def download_if_not_exist(remote_filename: str, local_path: str):
    """
    从 Modal Volume 下载文件到本地
    remote_filename: Volume 中的相对路径 (例如 "EqCmAYywrwI/baked.mkv")
    local_path: 本地保存路径 (例如 "./output/final.mkv")
    """
    local_file = Path(local_path)
    
    # 1. 检查本地是否已存在
    if local_file.exists():
        print(f"✅ 本地文件已存在，跳过下载: {local_path}")
        return True

    # 确保本地父目录存在
    local_file.parent.mkdir(parents=True, exist_ok=True)

    remote_file = remote_filename

    print(f"📥 开始下载: {remote_file} -> {local_path} ...")
    
    # 2. 调用 Modal CLI 下载
    # 格式: modal volume get <Volume名> <远程路径> <本地路径>
    cmd = [
        "modal", "volume", "get", 
        "data",           # 必须与你定义的 Volume 名称一致
        remote_file,      # 远程源文件
        str(local_file)       # 本地目标文件
    ]
    
    retry_count = 5
    for i in range(retry_count):
        try:
            # capture_output=True 会捕获 stdout/stderr，不让它刷屏
            # check=True 会在命令失败时抛出异常
            subprocess.run(cmd, check=True, text=True)
            print(f"🎉 下载成功: {local_path}")
            return True
        except subprocess.CalledProcessError as e:
            # 如果下载失败（例如远程文件不存在），打印错误信息
            print(f"❌ 下载失败: {e}, 重试 {i+1}/{retry_count}, sleep {2 ** i}s")
            sleep(2 ** i)
    return False

# 3. 这就是你想要的 upload 函数
def uploadIfNotExist(local_path: str, remote_filename: str):
    local_file = Path(local_path)
    if not local_file.exists():
        raise FileNotFoundError(f"本地文件未找到: {local_path}")

    print(f"🔍 检查云端文件: {remote_filename} ...")
    
    # 调用上面的云端函数进行检查
    exists, size_mb = check_file_exists.remote(remote_filename)
    
    if exists:
        print(f"✅ 文件已存在 ({size_mb:.2f} MB)，跳过上传。")
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

# --- 使用示例 ---
# if __name__ == "__main__":
#     # 定义你要处理的文件
#     local_file_path = "/Users/toddzheng/.ktv_control_plane/data/EqCmAYywrwI/original.wav"
#     local_video_path = "/Users/toddzheng/.ktv_control_plane/data/EqCmAYywrwI/video_only.mp4"

    
#     # 定义在云端 Volume 中的相对路径 (不要以 / 开头，也不要加 /data)
#     # 结果会存到: /data/EqCmAYywrwI/original.wav
#     remote_relative_path = "EqCmAYywrwI/original.wav"
#     original_relative_path = "EqCmAYywrwI/vocals.wav"
#     video_relative_path = "EqCmAYywrwI/video_only.mp4"

#     local_path = Path("/Users/toddzheng/.ktv_control_plane/data/EqCmAYywrwI/")
#     remote_path = "EqCmAYywrwI"
#     # 启动 Modal App 上下文
#     with app.run():
#         try:
#             # 1. 上传
#             # uploadIfNotExist(local_file_path, remote_relative_path)
#             # uploadIfNotExist(local_video_path, video_relative_path)
            
#             # # 2. 调用 GPU 函数
#             # print("🚀 调用远程 GPU 进行分离...")
#             # result = demucsFn.remote(remote_relative_path)
            
#             # print("\n✅ 处理完成！结果文件路径：")
#             # print(f"   🎤 人声: {result['vocals']}")
#             # print(f"   🎹 伴奏: {result['instrumental']}")
#             # result = transcribe.remote(original_relative_path)
#             # print(result)
#             vocal_file = "vocals.wav"
#             original_file = "original.wav"
#             instrumental_file = "instrumental.wav"
#             if not download_if_not_exist(f"{remote_path}/{vocal_file}", f"{local_path}/{vocal_file}"):
#                 raise FileNotFoundError(f"人声文件未找到: {remote_path}/{vocal_file}")
#             if not download_if_not_exist(f"{remote_path}/{original_file}", f"{local_path}/{original_file}"):
#                 raise FileNotFoundError(f"原始文件未找到: {remote_path}/{original_file}")
#             if not download_if_not_exist(f"{remote_path}/{instrumental_file}", f"{local_path}/{instrumental_file}"):
#                 raise FileNotFoundError(f"伴奏文件未找到: {remote_path}/{instrumental_file}")

            
#         except Exception as e:
#             print(f"\n❌ 发生错误: {e}")