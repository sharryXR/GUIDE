import base64
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import math
import random
from PIL import Image, ImageDraw
import re
import os
from pathlib import Path
import webvtt
import av
from io import BytesIO
import json
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from guide.action_annotation_prompt import generate_vlm_action_prompt
# from guide.model.load_model import load_model
import argparse
from dotenv import load_dotenv
load_dotenv()

llm1 = ChatOpenAI(
    model = "qwen3-vl-8b-thinking",
    temperature = 1.0,
    api_key=os.getenv("OPENAI_QWEN_API_KEY"),
    base_url=os.getenv("OPENAI_QWEN_BASE_URL")
)

llm2 = ChatOpenAI(
    model="qwen2.5-vl-32b-instruct",
    temperature=1.0,
    api_key=os.getenv("OPENAI_QWEN_API_KEY"),
    base_url=os.getenv("OPENAI_QWEN_BASE_URL"),
)

llm3 = AzureChatOpenAI(
        model = "gpt-4.1-2025-04-14",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_AZURE_ENDPOINT.openai.azure.com/"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version='2024-02-01',
        temperature=1.0
    )

llm4 = ChatOpenAI(
    model="gpt-5.1-2025-11-13",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm5 = ChatOpenAI(
    model="gpt-5.1-2025-11-13",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm6 = ChatOpenAI(
    model="gpt-5-mini-2025-08-07",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm7 = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm8 = ChatOpenAI(
    model="qwen3-vl-8b-instruct",
    temperature=1.0,
    api_key=os.getenv("OPENAI_QWEN_API_KEY"),
    base_url=os.getenv("OPENAI_QWEN_BASE_URL"),
)
llm9 = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=1.0,
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_API_URL")
)

#get llm from model name
def get_llm(model_name):
    if model_name == "qwen-plus":
        return llm1
    elif model_name == "qwen2.5-vl-32b-instruct-api":
        return llm2
    # elif model_name == "Qwen2.5-VL-32B-Instruct":
    #     llm = load_model(
    #         model_name="Qwen/Qwen2.5-VL-32B-Instruct",
    #         local_path="/Users/sharry/models/Qwen2.5-VL-32B-Instruct",
    #         cache_dir="/Users/sharry/models/Qwen2.5-VL-32B-Instruct",
    #         temperature=1.0,
    #     )
    #     return llm
    # elif model_name == "Qwen2.5-VL-7B-Instruct":
    #     llm = load_model(
    #         model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    #         local_path="/Users/sharry/models/Qwen2.5-VL-7B-Instruct",
    #         cache_dir="/Users/sharry/models/Qwen2.5-VL-7B-Instruct",
    #         temperature=1.0,
    #     )
    #     return llm
    # elif model_name == "Qwen2.5-VL-3B-Instruct":
    #     llm = load_model(
    #         model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    #         local_path="/Users/sharry/models/Qwen2.5-VL-3B-Instruct",
    #         cache_dir="/Users/sharry/models/Qwen2.5-VL-3B-Instruct",
    #         temperature=1.0,
    #     )
    #     return llm
    elif model_name == "gpt-5.1":
        return llm4
    elif model_name == "gpt-5":
        return llm5
    elif model_name == "gpt-5-mini-2025-08-07":
        return llm6
    elif model_name == "gpt-4.1-mini-2025-04-14":
        return llm7
    elif model_name == "qwen3vl-8b":
        return llm8
    elif model_name == "seed1.8":
        return llm9
    elif model_name == "qwen3":
        return llm1
    else:
        raise ValueError("Unsupported model name")
    
def consolidate_descriptions(base_dir, model_name=None):
    """
    Consolidate thought and action NLP descriptions from annotation files.
    """
    # #将model_name拼接到base_dir
    # if model_name:
    #     base_dir = os.path.join(base_dir, model_name)
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return
    output_dir = os.path.join(base_dir, "consolidated")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        if root == output_dir:
            continue
        print(f"Processing directory: {root}")
        #首先进行文件名排序
        files.sort()
        #首先进行_labeled_thought_and_nlp_descriptions.txt文件的拼接
        labeled_nlp_descriptions = []
        #记录前缀key_frame_000编号
        useful_ids = []
        step = 1
        for file in files:
            if file.endswith("_thou_and_action.txt"):
                # Extract the prefix from the filename from key_frame_000_inception_00:00:00.500_labeled_nlp_descriptions.txt
                number_match= re.search(r'key_frame_(\d+)_inception', file)
                number = number_match.group(1) if number_match else None
                if number:
                    useful_ids.append(number)
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    labeled_nlp_descriptions.append(f"step {step}: {f.read().strip()}")
                step += 1
        if labeled_nlp_descriptions:
            # Write the consolidated labeled_nlp_descriptions to a new file
            query= root.split("/")[-1]
            #取query最后一项
            query = query.split("\\")[-1] if "\\" in query else query.split("/")[-1]
            #以"~~"分割
            if "~~" in query:
                query = query.split("~~")[0]
            with open(os.path.join(output_dir, f"{query}_thou_and_action.txt"), 'w', encoding='utf-8') as f:
                f.write(f"TASK: {query}\n")
                f.write("\n".join(labeled_nlp_descriptions))
            print(f"Consolidated labeled_nlp_descriptions.txt written to {output_dir}")
        # input("Press Enter to continue...")
                    

def run_sum(web,query,model_name=None):
    query_dir = query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir.rstrip()
    if model_name:
        base_dir = f"./videos/{web}/{query_dir}/Labeled_{model_name}"
    else:
        base_dir = f"./videos/{web}/{query_dir}/Labeled"
    consolidate_descriptions(base_dir, model_name)

# def extract_vlm_response_parts(response_text: str) -> Dict[str, Any]:
#     """
#     Safely extract Thought, Meaningful, Actions, Action NLP Descriptions, and Thought and Action NLP Descriptions
#     from a possibly incomplete response.
    
#     Parameters:
#     response_text (str): The raw response from the VLM.
    
#     Returns:
#     Dict[str, Any]: Extracted parts (can be partial if response is truncated).
#     """
#     def extract_block(key: str) -> str:
#         pattern = rf'"{key}"\s*:\s*"((?:[^"\\\\]|\\\\.)*?)"'
#         match = re.search(pattern, response_text, re.DOTALL)
#         return match.group(1).strip() if match else ""
    
#     def extract_list(key: str) -> List[str]:
#         pattern = rf'"{key}"\s*:\s*\[((?:.|\n)*?)\]'
#         match = re.search(pattern, response_text, re.DOTALL)
#         if match:
#             raw_list = match.group(1)
#             items = re.findall(r'"((?:[^"\\\\]|\\\\.).*?)"', raw_list, re.DOTALL)
#             return [item.strip() for item in items]
#         return []
    
#     def extract_boolean(key: str) -> bool:
#         pattern = rf'"{key}"\s*:\s*(true|false)'
#         match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
#         if match:
#             return match.group(1).lower() == "true"
#         return False  # Default to false if not found
    
#     return {
#         "Thought": extract_block("Thought"),
#         "Meaningful": extract_boolean("Meaningful"),
#         "Actions": extract_list("Actions"),
#         "Action NLP Descriptions": extract_list("Action NLP Descriptions"),
#         "Thought and Action NLP Descriptions": extract_block("Thought and Action NLP Descriptions")
#     }
def extract_vlm_response_parts(response_text: str) -> Dict[str, Any]:
    """
    Safely extract Thought, Meaningful, Actions, Action NLP Descriptions, and Thought and Action NLP Descriptions
    from a possibly incomplete response.
    
    Parameters:
    response_text (str): The raw response from the VLM.
    
    Returns:
    Dict[str, Any]: Extracted parts (can be partial if response is truncated).
    """
    def extract_block(key: str) -> str:
        # 更复杂的模式，处理多行文本和转义字符
        # 先尝试找到键的起始位置
        pattern = rf'"{key}"\s*:\s*"'
        match = re.search(pattern, response_text)
        if not match:
            return ""
        
        # 从匹配位置开始，逐字符解析直到找到未转义的引号
        start_pos = match.end()
        pos = start_pos
        result = []
        
        max_iterations = len(response_text) * 2  # 防止死循环
        iteration_count = 0
        
        while pos < len(response_text):
            iteration_count += 1
            if iteration_count > max_iterations:
                # 防止死循环，强制退出
                break
                
            if response_text[pos] == '\\' and pos + 1 < len(response_text):
                # 处理转义字符
                next_char = response_text[pos + 1]
                if next_char == 'n':
                    result.append('\n')
                elif next_char == '"':
                    result.append('"')
                elif next_char == '\\':
                    result.append('\\')
                else:
                    result.append(response_text[pos:pos+2])
                pos += 2
            elif response_text[pos] == '"':
                # 找到结束引号
                break
            else:
                result.append(response_text[pos])
                pos += 1
        
        return ''.join(result).strip()
    
    def extract_list(key: str) -> List[str]:
        pattern = rf'"{key}"\s*:\s*\[((?:.|\n)*?)\]'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            raw_list = match.group(1)
            # 改进的正则表达式，更好地处理引号内的内容
            items = []
            item_pattern = r'"((?:[^"\\]|\\.)*)+"'
            for item_match in re.finditer(item_pattern, raw_list):
                item = item_match.group(1)
                # 处理转义字符
                item = item.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                items.append(item.strip())
            return items
        return []
    
    def extract_boolean(key: str) -> bool:
        pattern = rf'"{key}"\s*:\s*(true|false)'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return False  # Default to false if not found
    
    return {
        "Thought": extract_block("Thought"),
        "Meaningful": extract_boolean("Meaningful"),
        "Actions": extract_list("Actions"),
        "Action NLP Descriptions": extract_list("Action NLP Descriptions"),
        "Thought and Action NLP Descriptions": extract_block("Thought and Action NLP Descriptions")
    }
def get_task_and_thought(vtt_path,time_index):
    """从字幕文件获取 task (标题) 和 thought (对应段落的字幕内容)"""
    task = Path(vtt_path).stem  # 提取字幕文件名作为任务标题
    task = task.split("~~")[0]

    # 如果 time_index 是 None，直接返回任务标题和空内容
    if not time_index:
        print("No time index found.")
        return task, ""

    # 将 time_index 转换为毫秒（格式 'HH:MM:SS.mmm'）
    h, m, s_ms = time_index.split("_")
    s, ms = s_ms.split(".")
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    # 读取 .vtt 文件
    captions = list(webvtt.read(vtt_path))  # 转换为列表以便索引
    thought_parts = []

    # 找到匹配当前时间戳的字幕段
    current_idx = None
    for i, caption in enumerate(captions):
        start_h, start_m, start_s_ms = caption.start.split(":")
        start_s, start_ms = start_s_ms.split(".")
        start_ms_total = (int(start_h) * 3600 + int(start_m) * 60 + int(start_s)) * 1000 + int(start_ms)

        end_h, end_m, end_s_ms = caption.end.split(":")
        end_s, end_ms = end_s_ms.split(".")
        end_ms_total = (int(end_h) * 3600 + int(end_m) * 60 + int(end_s)) * 1000 + int(end_ms)

        if start_ms_total <= total_ms <= end_ms_total:
            current_idx = i
            break

    if current_idx is None:
        return task, task
    # 获取前一句、当前句、后一句
    # 前一句（如果存在）
    if current_idx > 0:
        thought_parts.append(captions[current_idx - 1].text)
    else:
        thought_parts.append("[No previous subtitle]")

    # 当前句
    thought_parts.append(captions[current_idx].text)

    # 后一句（如果存在）
    if current_idx < len(captions) - 1:
        thought_parts.append(captions[current_idx + 1].text)
    else:
        thought_parts.append("[No next subtitle]")

    # 拼接 thought，添加分隔符（可选）
    thought = " ".join(thought_parts)  # 用 ' | ' 分隔，也可改为 '\n' 或其他
     
    thought = task + "." + thought[:1000]
    # print(f"Task: {task}")
    # print(f"Thought: {thought}")
    return task, thought


def extract_video_info(video_path):
    """解析视频路径，提取操作名、视频文件名、片段索引"""
    parts = Path(video_path).parts
    operation = parts[-3]  # 操作名，例如 "Adding text boxes"
    video_name = parts[-1]  # 片段文件名，例如 "segment_3_4.mp4"
    video_base = parts[-2]  # 原始视频文件名，例如 "How to Create Text Box~~fvfgRmEkzqA"
    # segment_match = re.search(r"frame_(\d+)_", video_name)
    # segment_index = int(segment_match.group(1)) if segment_match else None
    # print(video_name)
    time_match = re.search(r"_(\d{2}_\d{2}_\d{2}\.\d{3}).", video_name)
    time_index = time_match.group(1) if time_match else None
    # print(time_index)

    return operation, video_base, time_index

def resize_image_if_needed(image, max_pixels=3000 * 28 * 28):
    """ 限制图片像素总数，并保持原始纵横比 """
    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        new_width = int(image.width * resize_factor)
        new_height = int(image.height * resize_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)  # 修复：使用LANCZOS替代ANTIALIAS
        # print(f"图片被缩放至: {new_width}x{new_height}，缩放因子: {resize_factor:.4f}")
    return image

def construct_prompt_desktop(task, thought):
    """构造用于 UI 交互的 prompt"""
    return f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Task: {task}
    
## Output Format
```\nThought: ...
Action: ...\n```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\
\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use Chinese in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{thought}
"""

def construct_prompt_mobile(task, thought):
    """构造用于 UI 交互的 prompt"""
    return f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in `Thought` part.

- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{thought}
"""



#封装成对外部函数
def run_action_annotation(web,query,client,model_name=None):
    """运行动作标注，返回True表示成功，False表示失败"""
    query_dir = query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir.rstrip()
    img_root = f"./videos/{web}/{query_dir}/OmniParser_Pic"
    subtitle_root = f"./videos/{web}/{query_dir}/audios_text"
    if model_name:
        action_root = f"./videos/{web}/{query_dir}/Labeled_{model_name}"
    else:
        action_root = f"./videos/{web}/{query_dir}/Labeled"
    # json_path = f"/mnt/buffer/shangzirui/gui_videos/class_json/classification_results_{web}.json"
    # with open(json_path, "r", encoding="utf-8") as json_file:
    #     prompt_mapping = json.load(json_file)  # 读取 JSON 并解析
    #至多处理2个目录
    i= 0
    for root, _, files in os.walk(img_root):
        if root == img_root:
            continue
        print(f"Processing directory: {root}")
        i += 1
        # if i > 2:
        #     print("已处理超过2个目录，停止处理。")
        #     break
        files = sorted([f for f in files if not f.startswith('.')])  # 排序 + 过滤隐藏文件
        # 判断当前目录下文件数是否超过限制,取前50个
        if len(files) > 50:
            print(f"Skipping {root} because it has more than 50 files")
            files = files[:50]
        
        for i,file in enumerate(files):
            # 输入连续两张图片，过滤最后一张
            # 过滤偶数文件
            if (i+1) % 2 == 0:
                continue
            if i+2 > len(files)-2:
                continue
            print(f"\n=== 处理文件 {i+1}/{len(files)}: {file} ===")  # 改进调试信息
            if file.startswith("key_frame") and file.endswith(".png"):
                img_path1 = os.path.join(root, file)
                img_path2 = os.path.join(root, files[i+2])
                txt_path1 = os.path.join(root, files[i+1])
                txt_path2 = os.path.join(root, files[i+3])
                print(f"  图片对: {file} <-> {files[i+2]}")

                operation1, video_base1, time_index1 = extract_video_info(img_path1)
                operation2, video_base2, time_index2 = extract_video_info(img_path2)
                #将模型名称拼接到视频文件名中
                action_path = os.path.join(action_root,video_base1)
                os.makedirs(action_path, exist_ok=True)
                #删除名称中key_frame_000_time_00_00_00.000_labeled的_time_00_00_00.000部分
                file_name= Path(file).stem
                file_name = file_name.split("_time_")[0]
                file_name+= "_labeled"
                action_file_path = os.path.join(action_path, f"{file_name}.txt")
                thought_file_path = os.path.join(action_path, f"{file_name}_thought.txt")
                nlp_descriptions_file_path = os.path.join(action_path, f"{file_name}_nlp_descriptions.txt")
                thought_and_action_nlp_descriptions_file_path = os.path.join(action_path, f"{file_name}_thou_and_action.txt")
                if os.path.exists(action_file_path):  
                    print(f"  跳过（已存在）: {action_file_path}")  # 添加调试
                    continue
                
                print(f"  开始标注: {file_name}")  # 添加调试
                vtt_path1 = os.path.join(subtitle_root, f"{video_base1}.vtt")
                vtt_path2 = os.path.join(subtitle_root, f"{video_base2}.vtt")
                
                if os.path.exists(vtt_path1):
                    task1, thought1 = get_task_and_thought(vtt_path1, time_index1)
                else:
                    print(f"    字幕文件不存在: {vtt_path1}")  # 添加调试
                    continue
                # if os.path.exists(vtt_path2):
                #     task2, thought2 = get_task_and_thought(vtt_path2, time_index2)
                # else:
                #     print(f"Subtitle file not found: {vtt_path2}")
                #     continue
                task =task1
                thought = thought1 
            
                print(f"  正在加载和处理图片...")
                try:
                    with Image.open(img_path1) as image:
                        print(f"    图片1尺寸: {image.width}x{image.height}")
                        image1 = resize_image_if_needed(image)
                        print(f"    图片1处理后尺寸: {image1.width}x{image1.height}")

                        buffered = BytesIO()
                        image1.save(buffered, format="PNG")
                        encoded_string1 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        print(f"    图片1编码完成，大小: {len(encoded_string1)} 字符")
                except Exception as e:
                    print(f"    ❌ 图片1读取失败: {type(e).__name__}: {e}")  # 添加调试
                    continue
                try:
                    with Image.open(img_path2) as image:
                        print(f"    图片2尺寸: {image.width}x{image.height}")
                        image2 = resize_image_if_needed(image)
                        print(f"    图片2处理后尺寸: {image2.width}x{image2.height}")

                        buffered = BytesIO()
                        image2.save(buffered, format="PNG")
                        encoded_string2 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        print(f"    图片2编码完成，大小: {len(encoded_string2)} 字符")
                except Exception as e:
                    print(f"    ❌ 图片2读取失败: {type(e).__name__}: {e}")  # 添加调试
                    continue

                # 读取 txt 文件
                with open(txt_path1, "r", encoding="utf-8") as f:
                    json_file_1 = f.read()
                with open(txt_path2, "r", encoding="utf-8") as f:
                    json_file_2 = f.read()
                # 合成 prompt
                prompt = generate_vlm_action_prompt(
                    json_file_1=json_file_1,
                    json_file_2=json_file_2,
                    task_description=task,
                    thought=thought
                )
                #截断json，避免过长
                # prompt = generate_vlm_action_prompt(
                #     json_file_1=json_file_1[0:1000],
                #     json_file_2=json_file_2[0:1000],
                #     task_description=task,
                #     thought=thought
                # )
                # print(f"prompt: {prompt}")
                # #prompt长度
                # prompt_length = len(prompt)
                # print(f"Prompt length: {prompt_length}")
                
                # print("task:",task)
                # print("thought:",thought)
                
                # 最多尝试2次完整的"API调用+解析"流程
                parse_success = False
                for attempt in range(2):
                    if attempt > 0:
                        print(f"\n  第 {attempt + 1} 次完整尝试（API调用+解析）...")
                    
                    success = False
                    for retry_idx in range(3):
                        import time
                        if retry_idx > 0:
                            time.sleep(5)
                        else:
                            time.sleep(3)
                        print(f"    尝试API调用 (第 {retry_idx + 1}/3 次)...")  # 添加重试日志
                        try:
                            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                            import signal
                            
                            # 使用线程池和超时机制
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(
                                    client.invoke,
                                    input=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": prompt},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string1}"}},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string2}"}},
                                            ],
                                        },
                                    ],
                                    # max_tokens=10000,
                                )
                                # 设置90秒超时
                                response = future.result(timeout=90)
                        except FuturesTimeoutError:
                            print(f"    API调用超时（90秒），正在重试...")
                            continue
                        except Exception as e:
                            print(f"    API调用失败: {type(e).__name__}: {str(e)}")
                            continue
                        else:
                            success = True
                            print(f"    API调用成功")
                            break

                    # Check if API call was successful
                    if not success:
                        print(f"  ❌ API调用3次重试后仍失败")
                        if attempt == 1:  # 第二次完整尝试也失败
                            print(f"\n{'='*60}")
                            print(f"错误：图片 {file} 经过2次完整尝试后仍然失败")
                            print(f"为避免浪费资源，自动停止后续所有标注工作")
                            print(f"请检查：")
                            print(f"  1. 网络连接是否正常")
                            print(f"  2. API密钥是否有效")
                            print(f"  3. 模型服务是否可用")
                            print(f"  4. 图片是否损坏或过大")
                            print(f"{'='*60}\n")
                            return False  # 返回失败状态
                        continue  # 进入下一次完整尝试

                    # Extract the action from the response
                    print(f"  正在解析模型响应...")
                    print(f"    响应内容长度: {len(response.content)} 字符")
                    print(f"    响应前200字符: {response.content[:200]}")
                    # print(f"Response: {response.content}")
                    # input("Press Enter to continue...")
                    
                    # 使用signal实现真正的超时（仅限Linux/Unix）
                    import time as time_module
                    import signal
                    
                    parse_start_time = time_module.time()
                    print(f"    开始解析时间: {time_module.strftime('%H:%M:%S')}")
                    
                    class TimeoutException(Exception):
                        pass
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutException("解析超时")
                    
                    try:
                        # 设置30秒超时信号
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)  # 30秒后触发SIGALRM
                        
                        result = extract_vlm_response_parts(response.content)
                        
                        signal.alarm(0)  # 取消alarm
                        parse_end_time = time_module.time()
                        print(f"  响应解析完成，耗时: {parse_end_time - parse_start_time:.2f}秒")
                        parse_success = True
                        break  # 解析成功，跳出外层循环
                    except TimeoutException:
                        signal.alarm(0)  # 取消alarm
                        parse_end_time = time_module.time()
                        print(f"  ❌ 响应解析超时（30秒），实际耗时: {parse_end_time - parse_start_time:.2f}秒")
                        print(f"    响应前500字符: {response.content[:500]}")
                        if attempt == 1:  # 第二次尝试也解析失败
                            print(f"  两次尝试均解析失败，跳过此图片对")
                        continue  # 进入下一次完整尝试
                    except Exception as e:
                        signal.alarm(0)  # 取消alarm
                        parse_end_time = time_module.time()
                        print(f"  ❌ 响应解析失败（耗时 {parse_end_time - parse_start_time:.2f}秒）: {type(e).__name__}: {str(e)}")
                        print(f"    响应前500字符: {response.content[:500]}")
                        if attempt == 1:  # 第二次尝试也解析失败
                            print(f"  两次尝试均解析失败，跳过此图片对")
                        continue  # 进入下一次完整尝试
                
                # 如果两次完整尝试后仍未成功解析
                if not parse_success:
                    continue  # 跳过当前图片，处理下一张
                    
                meaningful = result["Meaningful"]
                if not meaningful:
                    with open(action_file_path, "w", encoding="utf-8") as f:
                        f.write("Meaningful: False")
                    print(f"  ⚠ Meaningful: False，跳过")
                    continue
                thought= result["Thought"]
                actions = result["Actions"]
                action_nlp_descriptions = result["Action NLP Descriptions"]
                thought_and_action_nlp_descriptions = result["Thought and Action NLP Descriptions"]
                #将thought_and_action_nlp_descriptions中的换行符替换为" "
                thought_and_action_nlp_descriptions = thought_and_action_nlp_descriptions.replace("\n", " ")
                # print(f"Thought: {thought}")
                # print(f"Actions: {actions}")
                # print(f"Action NLP Descriptions: {action_nlp_descriptions}")
                # print(f"Thought and Action NLP Descriptions: {thought_and_action_nlp_descriptions}")
                # input("Press Enter to continue...")
                # 将结果写入文件
                print(f"  正在保存结果到文件...")
                with open(action_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(actions))
                with open(thought_file_path, "w", encoding="utf-8") as f:
                    f.write(thought)
                with open(nlp_descriptions_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(action_nlp_descriptions))
                with open(thought_and_action_nlp_descriptions_file_path,"w",encoding="utf-8") as f:
                    f.write(thought_and_action_nlp_descriptions)
                print(f"  ✅ 成功标注: {file_name}")
                # wait 避免请求过快
    
    return True  # 全部处理成功


def run_get_divided_action_annotation(web,query,client,model_name=None):
    query_dir = query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir.rstrip()
    if model_name:
        base_dir = f"./videos/{web}/{query_dir}/Labeled_{model_name}"
    else:
        base_dir = f"./videos/{web}/{query_dir}/Labeled"
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return
    output_dir = os.path.join(base_dir, "devided")
    input_dir = os.path.join(base_dir, "consolidated")
    os.makedirs(output_dir, exist_ok=True)
    output_dir1 = os.path.join(output_dir, "planning")
    os.makedirs(output_dir1, exist_ok=True)
    output_dir2 = os.path.join(output_dir, "grounding")
    os.makedirs(output_dir2, exist_ok=True)
    divide_by_llm(input_dir, output_dir1, output_dir2, client)

def generate_planning_prompt(task_description, annotation_file):
    return f"""
You are a Visual Language Model (VLM) tasked with extracting high-level PLANNING information from a step-by-step GUI video annotation file. 
Your job is to produce a structured summary of the execution plan in two distinct parts.

### Inputs
- Task Description: {task_description}
- Annotation File: {annotation_file}  # multi-step textual annotation of how the task was executed

### Output Requirements
The output must contain **exactly two sections**:

1. **Execution Flow & Planning**  
   - A coherent narrative (or numbered flow) describing the overall task plan.  
   - Include the logical order of steps, what the user is trying to achieve at each stage, and how the workflow progresses.  
   - This should read like a summary of the full procedure, not just a list of actions.

2. **Key Considerations**  
   - A concise list of the most important cautions, pitfalls, or principles to ensure correct execution.  
   - Each point should be 1–2 sentences maximum.  
   - Focus only on high-value insights.

### Guidelines
- Do not include low-level UI coordinates or click details unless essential.  
- Keep the Execution Flow section detailed but readable, showing the big picture.  
- Keep the Key Considerations section short and actionable, suitable as a checklist.  
- Do not include reasoning steps, internal thought processes, or references to screenshots/JSON diffs.  
- Output must always contain both sections.

---

### Example Output (for format reference)
**task**: How to format paragraphs in LibreOffice Writer 2024

**Execution Flow & Planning**  
To format paragraphs in LibreOffice Writer 2024, the process begins by selecting all target paragraphs to ensure later changes apply consistently. The user then adjusts the workspace by switching to the “View” tab, enabling helpers like formatting marks or guide lines, and setting the document to “Optimal View” for clear visibility. Once the workspace is prepared, paragraphs are reviewed for readability; long sections are split with line breaks, and unnecessary blank lines are removed. Next, the main formatting menus (“Home” and “Format”) are opened to access tools for alignment, spacing, and styling. For more advanced customization, detailed dialogs such as “Paragraph…” or “Detail…” are opened, where indentation, spacing, borders, drop caps, or transparency can be configured. Throughout the process, sample content may be added to test formatting changes, and formatting marks are toggled to ensure structural clarity. Finally, all adjustments are confirmed via the “OK” button in dialog windows, ensuring that every change is applied and the document appears professional and consistent.

**Key Considerations**  
- Always select all relevant paragraphs before applying global formatting changes.  
- Use formatting marks and view adjustments to reveal hidden breaks and spaces.  
- Insert sample text before applying advanced styles, so changes are visible.  
- Confirm settings with “OK” in each dialog to ensure they are applied.  
- Remove unnecessary blank lines to maintain professional spacing.  
- Use contextual menus for precise adjustments when formatting only part of the text.  
"""

def generate_grounding_prompt(task_description, annotation_file):
    return f"""
You are a Visual Language Model (VLM) tasked with extracting GROUNDING information from a step-by-step GUI video annotation file. 
Your goal is to identify interactive icons/controls described in the annotation, summarize how they appear and where they are located, 
and infer their most likely functions **along with possible task-related operations**.

### Inputs
- Task Description: {task_description}
- Annotation File: {annotation_file}  # multi-step textual annotation with descriptions of GUI elements and actions

### Output Requirements
The output must contain a list of identified interactive elements.  
For each element, provide:

1. **Icon/Control**  
   - The name or label of the interactive element, or a short identifier if no explicit name is given.

2. **Appearance & Position**  
   - A concise description of its visible properties (e.g., color, shape, text label) and relative position on screen (e.g., top-left toolbar, center of popup).

3. **Predicted Function & Related Task Operations**  
   - The most likely role or function of this element based on its description and context.  
   - Additionally, describe the **typical task operations** that might involve this element (e.g., "click to open a settings dialog," "toggle to reveal hidden formatting," "confirm by pressing after adjusting spacing").

### Guidelines
- Focus only on **interactive UI elements** (buttons, tabs, dialog options, icons).  
- Do not include passive text content unless it is clearly interactive.  
- Each element should be described independently.  
- Be concise and clear, avoiding internal reasoning or references to screenshots/JSON diffs.  
- Always provide all three fields: Icon/Control, Appearance & Position, Predicted Function & Related Task Operations.

---

### Example Output (for format reference)

1. Icon/Control:"View Tab"  
   - Appearance and Position: A labeled button “View” located in the top-left row of tabs, styled like “File” and “Insert.”  
   - Predicted Function and Related Task Operations: Switches the workspace into View mode. Likely used in the task to enable visual aids such as “Helplines While Moving” and adjust document display (e.g., selecting “Optimal View”).

2. Icon/Control: "Formatting Marks"  
   - Appearance and Position: A rectangular toolbar button near the top center-left, showing the pilcrow symbol in blue-gray.  
   - Predicted Function and Related Task Operations: Toggles visibility of hidden formatting marks (spaces, breaks). In the task, it helps users identify extra blank lines and verify paragraph structure before cleaning up spacing.

3. Icon/Control: "Detail…" button  
   - Appearance and Position: A small rectangular button labeled “Detail…” on a horizontal toolbar near the top-center.  
   - Predicted Function and Related Task Operations: Opens advanced paragraph formatting options (indents, spacing, alignment). In the task, it is clicked to bring up the detailed settings dialog before confirming paragraph adjustments.

4. Icon/Control: "OK button in Paragraph dialog"  
   - Appearance and Position: A medium gray-blue rectangular button labeled “OK,” located at the bottom right of the Paragraph settings popup.  
   - Predicted Function and Related Task Operations: Confirms and applies formatting changes. In the task, it is pressed after adjusting spacing/indentation or styling (drop caps, borders) to finalize and close the dialog.

Ensure your output follows this structured format for each identified element and don't output anything else.Ensure **Icon/Control:**, **Appearance and Position:**, and **Predicted Function and Related Task Operations:** are always included for each element and these keys names are exactly as specified.

Ensure the listed elements are the most relevant to the task described, and the number of identified elements is no more than 15.   
   """

def divide_by_llm(input_dir, output_dir1, output_dir2, client):
    """使用llm提取分割为两个文件planning与grounding"""
    print(f"Dividing annotations in {input_dir} into planning and grounding...")
    files= os.listdir(input_dir)
    for file in files:
        if file.endswith("_thou_and_action.txt"):
            input_file_path = os.path.join(input_dir, file)
            with open(input_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 提取任务描述
            task_description = ""
            lines = content.splitlines()
            if lines and lines[0].startswith("TASK:"):
                task_description = lines[0][len("TASK:"):].strip()
            # 提取 annotation_file 内容
            annotation_file = "\n".join(lines[1:]).strip()
            if not annotation_file:
                print(f"No annotation content in {input_file_path}, skipping.")
                continue
            planning_prompt = generate_planning_prompt(task_description, annotation_file)
            success = False
            for i in range(3):
                import time
                time.sleep(5)
                try:
                    response = client.invoke(
                        input=planning_prompt,
                        max_tokens=20000,
                    )
                except Exception as e:
                    print(e)
                    continue
                else:
                    success = True
                    break
            if not success:
                print(f"Failed to get planning response for {file} after 3 attempts")
                continue
            planning_content = response.content
            # print(f"Planning Response: {planning_content}")
            # input("Press Enter to continue...")
            # 将结果写入文件
            output_file_path1 = os.path.join(output_dir1, file.replace("_thou_and_action.txt", "_planning.txt"))
            planning_content = planning_content.replace("\r\n", "\n").replace("\r", "\n")
            with open(output_file_path1, "w", encoding="utf-8") as f:
                f.write(planning_content)
            grounding_prompt = generate_grounding_prompt(task_description, annotation_file)
            # print(f"Grounding Prompt: {grounding_prompt}")
            # input("Press Enter to continue...")
            success = False
            for i in range(3):
                import time
                time.sleep(5)
                try:
                    response = client.invoke(
                        input=grounding_prompt,
                        max_tokens=20000,
                    )
                except Exception as e:
                    print(e)
                    continue
                else:
                    success = True
                    break
            if not success:
                print(f"Failed to get grounding response for {file} after 3 attempts")
                continue
            grounding_content = response.content
            # print(f"Grounding Response: {grounding_content}")
            # input("Press Enter to continue...")
            grounding_content = grounding_content.replace("\r\n", "\n").replace("\r", "\n")
            output_file_path2 = os.path.join(output_dir2, file.replace("_thou_and_action.txt", "_grounding.txt"))
            with open(output_file_path2, "w", encoding="utf-8") as f:
                f.write(grounding_content)

# model_name = "qwen2.5-vl-32b-instruct-api"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Action Annotation with specified web and query.")
    parser.add_argument('--web', type=str, required=True, help='The web application to use.')
    parser.add_argument('--query', type=str, required=True, help='The query to process.')
    parser.add_argument('--model_name', type=str, default="gpt-5.1", help='The model name to use.')
    args = parser.parse_args()

    # model_name = args.model_name
    # web="libreoffice_writer"
    # query="How to set double line spacing"
    # model_name = "gpt-4o"
    web= args.web
    query= args.query
    model_name = args.model_name
    client = get_llm(model_name)
    
    # 检查 grounding 文件是否已存在
    query_dir = query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir.rstrip()
    if model_name:
        base_dir = f"./videos/{web}/{query_dir}/Labeled_{model_name}"
    else:
        base_dir = f"./videos/{web}/{query_dir}/Labeled"
    grounding_dir = os.path.join(base_dir, "devided", "grounding")
    
    # 统计视频数量
    video_dir = f"./videos/{web}/{query_dir}/video"
    video_count = 0
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        video_count = len(video_files)
        print(f"Found {video_count} video(s) in {video_dir}")
    else:
        print(f"Warning: Video directory {video_dir} does not exist")
    
    # 统计grounding文件数量
    grounding_count = 0
    if os.path.exists(grounding_dir):
        grounding_files = [f for f in os.listdir(grounding_dir) if f.endswith('_grounding.txt')]
        grounding_count = len(grounding_files)
        print(f"Found {grounding_count} grounding file(s) in {grounding_dir}")
    
    # 只有当grounding文件数量与视频数量相同时才跳过处理
    if video_count > 0 and grounding_count == video_count:
        print(f"All {video_count} video(s) have been processed (grounding files count matches), skipping all processing.")
    else:
        if grounding_count > 0:
            print(f"Incomplete processing detected: {grounding_count}/{video_count} videos processed, continuing...")
        # 运行标注，检查是否成功
        annotation_success = run_action_annotation(args.web, args.query, client, model_name)
        if not annotation_success:
            print("\n标注过程失败，停止后续处理（run_sum 和 run_get_divided_action_annotation）")
        else:
            # 标注成功，继续后续处理
            run_sum(args.web, args.query, model_name)
            run_get_divided_action_annotation(args.web, args.query, client, model_name)

    # run_get_divided_action_annotation(web, query, client)