import os
import re
import whisper
import subprocess
import logging
import os
# os.environ["PATH"] += os.pathsep + r"E:\ffmpeg\bin"
logging.basicConfig(level=logging.INFO)

def extract_audio_from_video(video_path, audio_path):

    command = [
        'ffmpeg', '-i', video_path,  # 输入文件
        '-vn',  # 不处理视频
        '-acodec', 'pcm_s16le',  # 音频编码格式
        '-ar', '16000',  # 设置采样率
        '-ac', '1',  # 单声道
        audio_path  # 输出音频文件
    ]

    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(f"音频已提取到: {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 错误: {e.stderr.decode()}")
        raise


def transcribe_with_whisper(audio_path, model):
    print(f"正在转录音频: {audio_path}")
    # 提取音频
    
    # audio_output_path = audio_path.replace(".mp4", ".wav")
    # try:
    #     extract_audio_from_video(audio_path, audio_output_path)
    #     print(f"音频已提取: {audio_output_path}")
    # except Exception as e:
    #     print(f"音频提取失败: {e}")
    #     return None
    # 使用 Whisper 进行转录
    result = model.transcribe(audio_path, word_timestamps=True)
    return result

def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def format_transcription(result):
    output = []
    for segment in result["segments"]:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"].strip()
        output.append(f"{start_time} --> {end_time}\n{text}")
    return "\n\n".join(output)

def txt_to_vtt(txt_path, vtt_path):
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
    with open(vtt_path, 'w', encoding='utf-8') as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for line in lines:
            vtt_file.write(line)

def parse_vtt(vtt_content):
    segments = []
    lines = vtt_content.strip().split('\n')
    i = 0
    while i < len(lines):
        if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', lines[i]):
            timestamp = lines[i]
            text = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text.append(lines[i].strip())
                i += 1
            segments.append((timestamp, ' '.join(text)))
        i += 1
    return segments

def merge_segments(segments):
    merged_segments = []
    i = 0
    while i < len(segments):
        start_time, text = segments[i]
        end_time = start_time.split(' --> ')[1]
        while i + 1 < len(segments) and not text.endswith(('.', '?', '!')):
            next_start_time, next_text = segments[i + 1]
            end_time = next_start_time.split(' --> ')[1]
            text += ' ' + next_text
            i += 1
        merged_segments.append((f"{start_time.split(' --> ')[0]} --> {end_time}", text))
        i += 1
    return merged_segments

def generate_vtt(segments):
    if not segments:
        return "WEBVTT\n"
    vtt_output = "WEBVTT\n\n"
    for timestamp, text in segments:
        vtt_output += f"{timestamp}\n{text}\n\n"
    return vtt_output.strip()

def process_audio_files(audio_base_path, output_base_path, model):
    for root, dirs, files in os.walk(audio_base_path):
        for file in files:
            if file.endswith(".mp4"):
                audio_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, audio_base_path)
                output_folder = os.path.join(output_base_path, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                output_txt_path = os.path.join(output_folder, file.replace(".mp4", ".txt"))
                output_vtt_path = os.path.join(output_folder, file.replace(".mp4", ".vtt"))
                optimized_vtt_path = os.path.join(output_folder, file.replace(".mp4", "_optimized.vtt"))
                if os.path.exists(output_txt_path):
                    print(f"跳过处理，文件已存在: {output_txt_path}")
                    continue
                ##使用绝对路径
                audio_file_path = os.path.abspath(audio_file_path)
                print(f"音频文件路径: {audio_file_path}")
                
                print(f"正在处理: {audio_file_path}")
                try:
                    transcription_result = transcribe_with_whisper(audio_file_path, model)
                except:
                    print(f"处理失败: {audio_file_path}")
                    continue
                if transcription_result is None:
                    print(f"转录失败: {audio_file_path}")
                    continue
                formatted_output = format_transcription(transcription_result)
                with open(output_txt_path, "w", encoding="utf-8") as output_file:
                    output_file.write(formatted_output)
                print(f"文本已保存: {output_txt_path}")
                
                txt_to_vtt(output_txt_path, output_vtt_path)
                print(f"VTT 已生成: {output_vtt_path}")
                
                vtt_content = open(output_vtt_path, 'r', encoding='utf-8').read()
                segments = parse_vtt(vtt_content)
                merged_segments = merge_segments(segments)
                final_vtt = generate_vtt(merged_segments)
                with open(optimized_vtt_path, "w", encoding="utf-8") as output_file:
                    output_file.write(final_vtt)
                print(f"优化 VTT 已保存: {optimized_vtt_path}")

#外部调用函数
def run_asr(web,query):
    query_dir= query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir[:-1]
    audio_base_path = f"./videos/{web}/{query_dir}/video"
    output_base_path = f"./videos/{web}/{query_dir}/audios_text"
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    print("加载 Whisper 模型...")
    model = whisper.load_model("base")
    # model = whisper.load_model("base", device="cuda")
    process_audio_files(audio_base_path, output_base_path, model)


# if __name__ == "__main__":
#     for web in ["Google", "Yandex", "Linkedin", "Reddit", "Telegram"]:
#         audio_base_path = f"./videos/{web}/videos"
#         output_base_path = f"./videos/{web}/audios_text"
#         print("加载 Whisper 模型...")
#         model = whisper.load_model("base")
#         process_audio_files(audio_base_path, output_base_path, model)

if __name__ == "__main__":
    # run_asr("Vscode", "Vscode安装python插件")
    run_asr("Microsoft Word", "insert table in Microsoft Word")