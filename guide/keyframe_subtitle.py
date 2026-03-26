import json
import av
import cv2
import numpy as np
import webvtt
import os
from skimage.metrics import structural_similarity as ssim
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor

# 将时分秒格式转换为秒数
def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split('.')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
    return total_seconds

# 读取字幕文件并提取时间段
def get_segments_from_subtitles(subtitle_file, video_duration):
    segments = []
    for caption in webvtt.read(subtitle_file):
        start_time = time_to_seconds(caption.start)
        end_time = time_to_seconds(caption.end)
        segments.append((start_time, end_time))
    
    if not segments:
        print(f"字幕文件 {subtitle_file} 为空，默认使用整个视频时长")
        return [(0, video_duration)]

    return segments

# 使用 PyAV 解码视频并提取帧
def process_video_segment(container, video_stream, fps, start_time, end_time, segment_index, output_folder, key_frame_data):
    # 初始化背景减法器
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # 参数设置
    threshold = 10000  # 差异像素数量的阈值
    key_frame_indices = []  # 用于保存所有检测到的关键帧的索引

    container.seek(int(start_time * av.time_base))  # 定位到 start_time 对应的时间戳

    i = 0

    for frame in container.decode(video_stream):
        timestamp = frame.time

        # 如果当前帧的时间在需要的时间段内，则处理
        if start_time <= timestamp <= end_time:
            # 获取前景掩码
            frame_image = frame.to_ndarray(format='bgr24')

            # 获取前景掩码
            fgmask = fgbg.apply(frame_image)

            # 对前景掩码进行阈值处理，去除噪声
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

            # 计算当前帧与背景的差异
            non_zero_count = cv2.countNonZero(thresh)

            if non_zero_count > threshold:
                key_frame_indices.append(i)
            i += 1

    # 处理关键帧索引列表：去掉不符合条件的关键帧
    n = 2
    final_key_frame_indices = []
    # 初始化连续序列的起点
    start_idx = key_frame_indices[0] if key_frame_indices else None  

    for i in range(1, len(key_frame_indices)):
        current_idx = key_frame_indices[i]
        prev_idx = key_frame_indices[i - 1]

        # 如果当前帧与前一帧的间隔大于 n，说明上一个连续段结束
        if current_idx - prev_idx > n:
            # 记录该连续序列的第一帧和最后一帧
            if start_idx is not None:
                final_key_frame_indices.append(start_idx)  # 记录起始帧
                if prev_idx != start_idx:
                    final_key_frame_indices.append(prev_idx)  # 记录结束帧
                else:
                    final_key_frame_indices.append(start_idx)  # 若只有一帧，保存两次
            start_idx = current_idx  # 更新新序列起点

    # 处理最后一个连续序列
    if start_idx is not None:
        final_key_frame_indices.append(start_idx)
        if key_frame_indices[-1] != start_idx:
            final_key_frame_indices.append(key_frame_indices[-1])
        else:
            final_key_frame_indices.append(start_idx)  # 若只有一帧，保存两次

    # 保存最终的关键帧数据
    i = 0
    if final_key_frame_indices:
        if start_time < (start_time + final_key_frame_indices[0] / fps):
            key_frame_data[(segment_index, 0)] = (start_time, start_time + final_key_frame_indices[0] / fps)  # 记录第一个关键帧
        i += 1

        while i < len(final_key_frame_indices) - 1:
            start_idx = final_key_frame_indices[i]
            end_idx = final_key_frame_indices[i + 1]
            key_frame_data[(segment_index, i // 2 + 1)] = (start_idx / fps + start_time, end_idx / fps + start_time)  # 只记录 idx
            i += 2  # 每次处理两个关键帧

        if i < len(final_key_frame_indices) and (final_key_frame_indices[-1] / fps + start_time) < end_time:
            key_frame_data[(segment_index, i // 2 + 1)] = (final_key_frame_indices[-1] / fps + start_time, end_time)
    else:
        if start_time < end_time:
            key_frame_data[(segment_index, 0)] = (start_time, end_time)

def cut_video(video_path, start_time, end_time, output_path):
    """
    截取视频的某一时间段并保存到新的文件中
    """
    
    # 使用 ffmpeg 命令行工具进行视频截取
    # command = [
    #     'ffmpeg',
    #     '-i', video_path,
    #     '-ss', str(start_time),
    #     '-to', str(end_time),
    #     '-c', 'copy',
    #     output_path
    # ]

    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', video_path,
        '-vframes', '1',
        output_path
    ]

    # command = [
    #     'ffmpeg',
    #     '-i', video_path,
    #     '-ss', str(start_time),
    #     '-to', str(end_time),
    #     '-c:v', 'libx264',
    #     '-c:a', 'aac',
    #     output_path
    # ]
    
    # 执行命令
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 主程序
def process_video(video_path, subtitle_file, output_folder, output_folder_v):

    json_filename = os.path.join(output_folder, f"key_frame.json")

    if not os.path.exists(json_filename):
        # 打开视频文件
        container = av.open(video_path)

        # 获取视频总时长
        video_duration = container.duration / av.time_base  # 视频总时长，单位为秒
        if video_duration > 600:
            # print(f"视频 {video_path} 超过600秒，跳过处理。")
            container.close()
            return
        
        segments = get_segments_from_subtitles(subtitle_file, video_duration)

        # 创建对应的关键帧输出目录
        os.makedirs(output_folder, exist_ok=True)

        video_stream = next(s for s in container.streams if s.type == 'video')
        fps = video_stream.average_rate

        key_frame_data = {}

        # 逐个处理每个字幕段落
        for segment_index, (start_time, end_time) in enumerate(segments):
            if segment_index < len(segments) - 1:
                end_time = segments[segment_index+1][0]
            else:
                end_time = video_duration
            print(f"处理第 {segment_index + 1} 段: 从 {start_time} 到 {end_time}")
            process_video_segment(container, video_stream, fps, start_time, end_time, segment_index + 1, output_folder, key_frame_data)

        container.close()

        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump({str(k): v for k, v in key_frame_data.items()}, json_file, ensure_ascii=False, indent=4)
    else:
        # 打开视频文件
        container = av.open(video_path)

        # 获取视频总时长
        video_duration = container.duration / av.time_base  # 视频总时长，单位为秒

        container.close()

        # 读取已存在的 JSON 文件，加载 key_frame_data
        with open(json_filename, "r", encoding="utf-8") as json_file:
            key_frame_data = json.load(json_file)
        key_frame_data = {eval(k): v for k, v in key_frame_data.items()}
    
    sorted_keys = sorted(key_frame_data.keys(), key=lambda x: x[0])  # 按片段索引排序

    for i, keyframe_key in enumerate(sorted_keys):
        segment_index, key_frame_index = keyframe_key  # 解析 keyframe 键

        segment_output_path = os.path.join(output_folder_v, f"segment_{segment_index}_{key_frame_index}.png")
        
        if os.path.exists(segment_output_path):
            continue

        start_time = key_frame_data[keyframe_key][0]
        end_time = key_frame_data[keyframe_key][1]

        # 检查是否超出视频时长
        if start_time >= video_duration:
            print(f"跳过片段 {segment_index}_{key_frame_index}，因为开始时间超出了视频长度")
            break
        
        if end_time > video_duration:
            end_time = video_duration  # 限制最大值

        # 处理 (start_time, end_time) 的视频切片
        print(f"正在分割片段: {segment_output_path}, 时间: {start_time} - {end_time}")
        cut_video(video_path, start_time, end_time, segment_output_path)


# 获取所有子文件夹和文件
def get_subtitle_video_pairs(subtitle_base_path, video_base_path):
    pairs = []  # 用于存储字幕文件和对应视频文件的路径
    for root, dirs, files in os.walk(subtitle_base_path):
        for file in files:
            # 只处理 .vtt 文件
            if file.endswith('.vtt'):
                subtitle_file = os.path.join(root, file)

                # 提取操作子文件夹名称和文件名
                relative_path = os.path.relpath(root, subtitle_base_path)
                video_folder = os.path.join(video_base_path, relative_path)

                # 提取字幕文件名对应的视频名
                video_name = file.replace('.vtt', '.mp4')  # 假设字幕后缀为 `.en.vtt`
                video_file = os.path.join(video_folder, video_name)

                # 检查视频文件是否存在
                if os.path.exists(video_file):
                    pairs.append((subtitle_file, video_file))
                else:
                    print(f"警告: 未找到对应的视频文件: {video_file}")

    return pairs


# 修改主程序入口
def process_all_videos_and_subtitles_parallel(subtitle_base_path, video_base_path, keyframes_base_path, output_base_path):
    # 获取字幕文件和视频文件的匹配对
    pairs = get_subtitle_video_pairs(subtitle_base_path, video_base_path)

    # 获取服务器的 CPU 核心数
    cpu_cores = os.cpu_count()
    print(f"当前服务器的 CPU 核心数：{cpu_cores}")

    # 使用 ProcessPoolExecutor 启动多进程
    with ProcessPoolExecutor(max_workers=int(3/4*cpu_cores)) as executor:
        # 对每对字幕视频进行处理
        for subtitle_file, video_file in pairs:
            print(f"开始处理视频: {video_file} 和字幕: {subtitle_file}")
            # 生成关键帧保存路径
            relative_path = os.path.relpath(video_file, video_base_path)

            output_folder = os.path.join(keyframes_base_path, os.path.dirname(relative_path), os.path.basename(video_file).replace('.mp4', ''))
            os.makedirs(output_folder, exist_ok=True)

            output_folder_v = os.path.join(output_base_path, os.path.dirname(relative_path), os.path.basename(video_file).replace('.mp4', ''))
            os.makedirs(output_folder_v, exist_ok=True)

            # 提交任务到进程池
            executor.submit(process_video, video_file, subtitle_file, output_folder, output_folder_v)
#外部调用函数
def run_keyframe_subtitle(web,query):
    video_base_path = f"./videos/{web}/{query}/video"
    subtitle_base_path = f'./videos/{web}/{query}/audios_text'
    keyframes_base_path = f'./videos/{web}/{query}/keyframes_subtitle'
    output_base_path =f'./videos/{web}/{query}/final_images'
    process_all_videos_and_subtitles_parallel(subtitle_base_path, video_base_path, keyframes_base_path, output_base_path)
    

# 调用并行处理函数
if __name__ == "__main__":
    run_keyframe_subtitle('Vscode', 'Vscode安装插件')