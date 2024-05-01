import os
import whisperx
import subprocess
import torch
import opencc
import time

# FFmpeg 環境變數
ffmpeg_path = 'C:\\Users\\aa920\\anaconda3\\envs\\test_env\\Library\\bin\\ffmpeg.exe'
os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = whisperx.load_model("large", device=device)

converter = opencc.OpenCC('s2t.json')

py_path = "C:\\Users\\aa920\\anaconda3\\envs\\nlp-text-emotion\\python.exe"
process = subprocess.Popen(
    [py_path, 'PB.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

print("wait 10 sec !!!")
time.sleep(10) # 等待PB載入模型

for i in range(1, 4):

    audio_path = str(i) + ".mp3"
    print(f"reading {audio_path}\n")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language="zh")
    text = ""

    for segment in result['segments']:
        text = text + segment['text'] + " "

    text = converter.convert(text)
    
    process.stdin.write(text + "\n")
    process.stdin.flush()

    response = process.stdout.readline().strip()

    print(f"text: {text}\t emotion: {response}")