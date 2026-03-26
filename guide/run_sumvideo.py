import cv2
import os
import subprocess
import json

def run_sumvideo(web,query):
    print(f"开始处理 {web}:{query} 的视频文件sumvideo...")
    query_dir = query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir[:-1]
    video_base_path = f"./videos/{web}/{query_dir}/video"
    keyframes_base_path = f'./videos/{web}/{query_dir}/keyframes_sumvideo_time'
    for video in os.listdir(video_base_path):
        video_path = os.path.join(video_base_path,video)
        output_dir = os.path.join(keyframes_base_path,os.path.basename(video_path).replace('.mp4', ''))
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"已存在输出目录且不为空: {output_dir}，跳过处理。")
            continue
        os.makedirs(output_dir, exist_ok=True)  # 创建目录，如果不存在
        
        # 设置参数
        nframes = 15  # 提取 15 个关键帧
        width = 1920  # 图像宽度
        height = 1080  # 图像高度

        # 使用 ffprobe 获取视频信息
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,r_frame_rate,nb_frames',
                '-of', 'json',
                video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            
            # 获取编码器和帧率
            stream_info = probe_data['streams'][0]
            codec_name = stream_info.get('codec_name', 'unknown')
            fps_str = stream_info.get('r_frame_rate', '30/1')
            fps = eval(fps_str)  # 转换 "25/1" 为 25.0
            
            # 获取总帧数
            if 'nb_frames' in stream_info:
                total_frames = int(stream_info['nb_frames'])
            else:
                # 如果没有 nb_frames，尝试用 ffprobe 计算
                count_cmd = [
                    'ffprobe', '-v', 'error',
                    '-count_frames',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_read_frames',
                    '-of', 'json',
                    video_path
                ]
                count_result = subprocess.run(count_cmd, capture_output=True, text=True, check=True)
                count_data = json.loads(count_result.stdout)
                total_frames = int(count_data['streams'][0]['nb_read_frames'])
            
            print(f"视频编码: {codec_name}, FPS: {fps}, 总帧数: {total_frames}")
            
        except Exception as e:
            print(f"获取视频信息失败: {video_path}, 错误: {e}")
            continue

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息（作为备份）
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        cv_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果 ffprobe 获取失败，使用 OpenCV 的值
        if fps <= 0:
            fps = cv_fps if cv_fps > 0 else 30
        if total_frames <= 0:
            total_frames = cv_total_frames
        
        if total_frames <= 0:
            print(f"无法获取视频帧数: {video_path}")
            cap.release()
            continue
        
        # 计算等时间间隔的帧索引
        # 使用与videosum time算法相同的方式：均匀分布
        frame_indices = []
        if nframes == 1:
            frame_indices = [total_frames // 2]
        else:
            interval = total_frames / nframes
            frame_indices = [int(i * interval) for i in range(nframes)]
        
        # 提取关键帧
        key_frames = []
        
        # 检查是否是 AV1 编码，如果是则使用 ffmpeg
        if codec_name == 'av1':
            print(f"检测到 AV1 编码，使用 ffmpeg 提取关键帧")
            cap.release()  # 释放 OpenCV 的句柄
            
            # 使用 ffmpeg 提取指定帧
            for frame_index in frame_indices:
                timestamp = frame_index / fps
                
                # 使用 ffmpeg 提取特定时间戳的帧
                temp_frame_path = os.path.join(output_dir, f"temp_frame_{frame_index}.jpg")
                ffmpeg_cmd = [
                    'ffmpeg', '-ss', str(timestamp),
                    '-i', video_path,
                    '-vf', f'scale={width}:{height}',
                    '-frames:v', '1',
                    '-y',  # 覆盖已存在的文件
                    temp_frame_path
                ]
                
                try:
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    # 读取提取的帧
                    frame = cv2.imread(temp_frame_path)
                    if frame is not None:
                        key_frames.append((frame, frame_index))
                    # 删除临时文件
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                except subprocess.CalledProcessError as e:
                    print(f"ffmpeg 提取帧 {frame_index} 失败: {e}")
                    continue
        else:
            # 使用 OpenCV 提取帧（适用于非 AV1 编码）
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    # 调整尺寸
                    resized_frame = cv2.resize(frame, (width, height))
                    key_frames.append((resized_frame, frame_index))
            
            cap.release()
        
        # 保存关键帧并计算时间戳（保持与原代码相同的格式）
        for i, (frame, frame_index) in enumerate(key_frames):
            # 计算时间戳（秒，带小数）
            timestamp = frame_index / fps
            # 转换为毫秒
            total_milliseconds = int(timestamp * 1000)
            hours = total_milliseconds // 3600000
            minutes = (total_milliseconds % 3600000) // 60000
            seconds = (total_milliseconds % 60000) // 1000
            milliseconds = total_milliseconds % 1000
            # 格式化为 HH:MM:SS.mmm
            time_str = f"{hours:02d}_{minutes:02d}_{seconds:02d}.{milliseconds:03d}"
            
            # 保存关键帧
            output_path = os.path.join(output_dir, f"key_frame_{i:03d}_time_{time_str}.jpg")
            cv2.imwrite(output_path, frame)
        
        print(f"完成处理: {video_path}，提取了 {len(key_frames)} 个关键帧")

def test_single_video(video_path, output_base_dir):
    """测试单个视频的关键帧提取"""
    print(f"开始测试视频: {video_path}")
    
    # 创建输出目录
    video_name = os.path.basename(video_path).replace('.mp4', '')
    output_dir = os.path.join(output_base_dir, video_name)
    
    if os.path.exists(output_dir):
        print(f"输出目录已存在: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
        print(f"已清空目录")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置参数
    nframes = 15  # 提取 15 个关键帧
    width = 1920  # 图像宽度
    height = 1080  # 图像高度

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率（每秒帧数）
    if fps <= 0:  # 如果 FPS 获取失败，手动指定
        fps = 30  # 默认值
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    
    print(f"视频信息: FPS={fps}, 总帧数={total_frames}")
    
    if total_frames <= 0:
        print(f"无法获取视频帧数: {video_path}")
        cap.release()
        return
    
    # 计算等时间间隔的帧索引
    frame_indices = []
    if nframes == 1:
        frame_indices = [total_frames // 2]
    else:
        interval = total_frames / nframes
        frame_indices = [int(i * interval) for i in range(nframes)]
    
    print(f"帧索引: {frame_indices}")
    
    # 提取关键帧
    key_frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # 调整尺寸
            resized_frame = cv2.resize(frame, (width, height))
            key_frames.append((resized_frame, frame_index))
    
    cap.release()
    
    # 保存关键帧并计算时间戳
    for i, (frame, frame_index) in enumerate(key_frames):
        # 计算时间戳（秒，带小数）
        timestamp = frame_index / fps
        # 转换为毫秒
        total_milliseconds = int(timestamp * 1000)
        hours = total_milliseconds // 3600000
        minutes = (total_milliseconds % 3600000) // 60000
        seconds = (total_milliseconds % 60000) // 1000
        milliseconds = total_milliseconds % 1000
        # 格式化为 HH:MM:SS.mmm
        time_str = f"{hours:02d}_{minutes:02d}_{seconds:02d}.{milliseconds:03d}"
        
        # 保存关键帧
        output_path = os.path.join(output_dir, f"key_frame_{i:03d}_time_{time_str}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"保存: {output_path}")
    
    print(f"完成！提取了 {len(key_frames)} 个关键帧到: {output_dir}")

if __name__ == '__main__':
    # 测试单个视频
    test_video = "./videos/example_video.mp4"
    test_output = "./test_keyframes_output"
    test_single_video(test_video, test_output)
    
    # 正常运行使用下面这行
    # run_sumvideo("Vscode","Vscode安装插件")