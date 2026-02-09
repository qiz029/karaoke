# KTV Song Processor

This project automates the creation of KTV-style karaoke videos from YouTube sources. It orchestrates a pipeline that downloads, splits, processes, and subtitles music videos.

## Features

- **Download**: Fetches high-quality audio and video from YouTube.
- **Split**: Separates raw media into video and audio tracks.
- **Demucs**: Uses AI to separate vocals from the instrumental backing track.
- **Transcribe**: Generates timestamped karaoke subtitles (ASS format) using `stable-ts` (Whisper).

## Dependencies

### System Requirements

- **FFmpeg**: Required for media processing (splitting and format conversion).
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Python Packages

The project requires the following Python libraries:

- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**: A command-line program to download videos from YouTube.
- **[demucs](https://github.com/facebookresearch/demucs)**: A state-of-the-art music source separation model.
- **[stable-ts](https://github.com/jianfch/stable-ts)**: A library for modifying OpenAI's Whisper to produce more accurate timestamps for subtitles.

## Installation

1. Ensure you have Python installed (3.8+ recommended).
2. Install the required Python packages using `pip`:

   ```bash
   pip install -r requirement.txt
   ```

   *Note: `demucs` and `stable-ts` may require PyTorch. If not installed automatically, please visit [pytorch.org](https://pytorch.org/) to install the version appropriate for your system.*

## Usage

1. Open `orchestration.py` and update the `yt_token` variable in the `__main__` block with the YouTube video ID you want to process.

   ```python
   if __name__ == "__main__":
       yt_token = "BKld0fxCu9k"  # Replace with your YouTube ID
       processor = YtSongProcessor(yt_token)
       processor.run()
   ```

2. Run the script:

   ```bash
   python orchestration.py
   ```

## Output

The script organizes output in the `data/<yt_token>/` directory:
- `raw.wav`: Original downloaded audio.
- `video_only.mp4`: Video track without audio.
- `original.wav`: Extracted PCM audio.
- `instrumental.wav`: Backing track (no vocals).
- `vocals.wav`: Extracted vocals.
- `lyrics.ass`: Generated KTV subtitles.
