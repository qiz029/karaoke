# 使用 slim 版本避免 glibc 兼容性问题
FROM python:3.11-slim

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 设置工作目录，后续命令都会在此目录下执行
WORKDIR /app

# 1. 安装系统级依赖 (ffmpeg 是 demucs 的核心依赖)
# 2. 清理缓存以缩小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \ 
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装 Python 依赖
# 建议将依赖写在 requirements.txt 中，这里为了方便直接安装
RUN pip install --no-cache-dir -U demucs stable-ts yt-dlp yt_dlp syncedlyrics

RUN pip install fastapi[standard] uvicorn asyncio

RUN pip install torch==2.5.1 torchaudio==2.5.1 soundfile faster-whisper

# 拷贝代码
COPY orchestration.py .
COPY main.py .

EXPOSE 8080

# 执行
CMD ["python", "main.py"]
