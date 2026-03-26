import subprocess
import os
import json

def run_cmd1_only(web, query):
    """只执行cmd1：音频转录和关键帧提取"""
    print(f"[CMD1] Processing {web}: {query}")
    conda_base = os.path.expanduser("~/anaconda3")
    env = os.environ.copy()
    env['VIDEO_WEB'] = web
    env['VIDEO_QUERY'] = query
    
    cmd1 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate video_self_learning && ' \
           f'cd ../guide && ' \
           f'python auto_catch.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY"'
    
    try:
        result = subprocess.run(cmd1, shell=True, executable='/bin/bash', env=env, timeout=600)
        if result.returncode == 0:
            print(f"✓ [CMD1] Completed successfully")
            return True
        else:
            print(f"✗ [CMD1] Failed with exit code: {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ [CMD1] Timed out")
        return False

def run_cmd2_only(web, query):
    """只执行cmd2：OmniParser UI解析"""
    print(f"[CMD2] Processing {web}: {query}")
    conda_base = os.path.expanduser("~/anaconda3")
    env = os.environ.copy()
    env['VIDEO_WEB'] = web
    env['VIDEO_QUERY'] = query
    
    cmd2 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate omni && ' \
           f'cd ../guide/OmniParser && ' \
           f'python auto_omni.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY"'
    
    try:
        result = subprocess.run(cmd2, shell=True, executable='/bin/bash', env=env, timeout=1800)
        if result.returncode == 0:
            print(f"✓ [CMD2] Completed successfully")
            return True
        else:
            print(f"✗ [CMD2] Failed with exit code: {result.returncode}")
            if result.returncode == -9 or result.returncode == 137:
                print(f"  (Process killed by system - likely OOM)")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ [CMD2] Timed out")
        return False

def run_cmd3_only(web, query, model="gpt-5.1"):
    """只执行cmd3：动作标注和轨迹生成，返回结果"""
    print(f"[CMD3] Processing {web}: {query}")
    conda_base = os.path.expanduser("~/anaconda3")
    env = os.environ.copy()
    env['VIDEO_WEB'] = web
    env['VIDEO_QUERY'] = query
    env['VIDEO_MODEL'] = model
    env['PYTHONUNBUFFERED'] = '1'

    cmd3 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate video_self_learning && ' \
           f'cd ../guide && ' \
           f'python -u action_annotation4.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY" --model_name "$VIDEO_MODEL"'
    
    try:
        result = subprocess.run(cmd3, shell=True, executable='/bin/bash', env=env, timeout=1200)
        if result.returncode != 0:
            print(f"✗ [CMD3] Failed with exit code: {result.returncode}")
            return None
        
        print(f"✓ [CMD3] Completed successfully")
        
        # 读取结果
        query_dir = query[:30].rstrip() if len(query) > 30 else query
        output_dir = f"../guide/videos/{web}/{query_dir}/Labeled_{model}/devided"
        grounding_dir = os.path.join(output_dir, "grounding")
        planning_dir = os.path.join(output_dir, "planning")
        
        grounding_results = ""
        planning_results = ""
        
        if os.path.exists(grounding_dir):
            for filename in os.listdir(grounding_dir):
                if filename.endswith('_grounding.txt'):
                    with open(os.path.join(grounding_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            task_name = filename.replace('_grounding.txt', '').split('~~')[0]
                            grounding_results += f"The grounding trajectory of Demo {len(grounding_results.split('The grounding trajectory'))}: {task_name}:\n{content}\n"
        
        if os.path.exists(planning_dir):
            for filename in os.listdir(planning_dir):
                if filename.endswith('_planning.txt'):
                    with open(os.path.join(planning_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            task_name = filename.replace('_planning.txt', '').split('~~')[0]
                            planning_results += f"The planning trajectory of Demo {len(planning_results.split('The planning trajectory'))}: {task_name}:\n{content}\n"
        
        return planning_results, grounding_results
    except subprocess.TimeoutExpired:
        print(f"✗ [CMD3] Timed out")
        return None

def auto_convert(web, query,model="gpt-5.1"):
    """
    自动转换函数，接收 web 和 query 参数。
    """
    print(f"Converting {web} with query: {query}")
    
    # 获取 conda 路径（通常在 ~/anaconda3 或 ~/miniconda3）
    conda_base = os.path.expanduser("~/anaconda3")  # 或 ~/miniconda3，根据实际情况修改
    
    # 通过环境变量传递参数，避免shell转义问题
    env = os.environ.copy()
    env['VIDEO_WEB'] = web
    env['VIDEO_QUERY'] = query
    
    # 脚本1：auto_catch.py 运行在 video_self_learning 环境（音频转录和关键帧提取）
    cmd1 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate video_self_learning && ' \
           f'cd ../guide && ' \
           f'python auto_catch.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY"'
    
    # 脚本2：auto_omni.py 运行在 omni 环境（UI元素解析）
    cmd2 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate omni && ' \
           f'cd ../guide/OmniParser && ' \
           f'python auto_omni.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY"'
    
    # 脚本3：action_annotation4.py 进行动作标注（轨迹生成）
    cmd3 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate video_self_learning && ' \
           f'cd ../guide && ' \
           f'python action_annotation4.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY"'
    
    # 分步执行，记录每步状态
    step_status = {"cmd1": False, "cmd2": False, "cmd3": False}
    
    # 步骤1：音频转录和关键帧提取
    print("=" * 60)
    print("Step 1/3: Running auto_catch.py (audio transcription & keyframe extraction)")
    try:
        result1 = subprocess.run(cmd1, shell=True, executable='/bin/bash', env=env, timeout=600)
        if result1.returncode == 0:
            print("✓ Step 1 completed successfully")
            step_status["cmd1"] = True
        else:
            print(f"✗ Step 1 failed with exit code: {result1.returncode}")
            raise Exception(f"auto_catch.py failed with exit code: {result1.returncode}")
    except subprocess.TimeoutExpired:
        print("✗ Step 1 timed out (10 minutes)")
        raise Exception("auto_catch.py timeout")
    
    # 步骤2：UI元素解析（可能因OOM失败）
    print("=" * 60)
    print("Step 2/3: Running auto_omni.py (OmniParser UI parsing)")
    try:
        result2 = subprocess.run(cmd2, shell=True, executable='/bin/bash', env=env, timeout=900)
        if result2.returncode == 0:
            print("✓ Step 2 completed successfully")
            step_status["cmd2"] = True
        else:
            print(f"✗ Step 2 failed with exit code: {result2.returncode}")
            if result2.returncode == -9 or result2.returncode == 137:
                print("  (Process killed by system - likely OOM)")
            raise Exception(f"auto_omni.py failed with exit code: {result2.returncode}")
    except subprocess.TimeoutExpired:
        print("✗ Step 2 timed out (15 minutes)")
        raise Exception("auto_omni.py timeout")
    
    # 步骤3：动作标注和轨迹生成
    print("=" * 60)
    print("Step 3/3: Running action_annotation4.py (action annotation & trajectory generation)")
    try:
        result3 = subprocess.run(cmd3, shell=True, executable='/bin/bash', env=env, timeout=600)
        if result3.returncode == 0:
            print("✓ Step 3 completed successfully")
            step_status["cmd3"] = True
        else:
            print(f"✗ Step 3 failed with exit code: {result3.returncode}")
            raise Exception(f"action_annotation4.py failed with exit code: {result3.returncode}")
    except subprocess.TimeoutExpired:
        print("✗ Step 3 timed out (10 minutes)")
        raise Exception("action_annotation4.py timeout")
    
    # 汇总状态
    print("=" * 60)
    print(f"Pipeline summary: cmd1={step_status['cmd1']}, cmd2={step_status['cmd2']}, cmd3={step_status['cmd3']}")
    if all(step_status.values()):
        print("✓ All steps completed successfully")
    else:
        failed_steps = [k for k, v in step_status.items() if not v]
        raise Exception(f"Pipeline incomplete. Failed steps: {', '.join(failed_steps)}")
    
    # 读取最终的标注结果并返回
    query_dir = query
    if len(query_dir) > 30:
        query_dir = query[:30]
    if query_dir.endswith(" "):
        query_dir = query_dir.rstrip()
    output_dir = f"../guide/videos/{web}/{query_dir}/Labeled_{model}/devided"
    grounding_dir = os.path.join(output_dir, "grounding")
    planning_dir = os.path.join(output_dir, "planning")

    grounding_results = ""
    planning_results = ""
    if not os.path.exists(grounding_dir):
        print(f"Warning: Grounding directory does not exist: {grounding_dir}")
    else:
        i=0
        for filename in os.listdir(grounding_dir):
            if filename.endswith('_grounding.txt'):
                print(f"Reading grounding file: {filename}")   
                file_path = os.path.join(grounding_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:
                        i += 1
                        task_name = filename.replace('_grounding.txt', '')
                        # 取~~前面的部分作为任务名
                        task_name = task_name.split('~~')[0]
                        grounding_results += (f"The grounding trajectory of Demo {i}: {task_name}:\n" + content + "\n")

    if not os.path.exists(planning_dir):
        print(f"Warning: Planning directory does not exist: {planning_dir}")
    else:
        j=0
        for filename in os.listdir(planning_dir):
            if filename.endswith('_planning.txt'):
                file_path = os.path.join(planning_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:
                        j += 1
                        task_name = filename.replace('_planning.txt', '')
                        # 取~~前面的部分作为任务名
                        task_name = task_name.split('~~')[0]
                        planning_results += (f"The planning trajectory of Demo {j}: {task_name}:\n" + content + "\n")
    return planning_results, grounding_results
    # # 读取 output_dir 下的所有文件
    # results = ""
    
    # if not os.path.exists(output_dir):
    #     print(f"Warning: Output directory does not exist: {output_dir}")
    #     return results
    
    # i = 0
    # for filename in os.listdir(output_dir):
    #     if filename.endswith('.txt'):
    #         file_path = os.path.join(output_dir, filename)
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             content = file.read().strip()
    #             if content:
    #                 i += 1
    #                 task_name = filename.replace('.txt', '')
    #                 # 取~~前面的部分作为任务名
    #                 task_name = task_name.split('~~')[0]
    #                 results += (f"The interaction trajectory of Demo {i}: {task_name}:\n" + content + "\n")
    
    # return results


def auto_download(web, query):
    """
    运行 ../guide/youtube.py 的下载函数。
    
    参数:
    web: 应用名称
    query: 查询语句
    
    返回:
    (video_count, stdout, stderr) 元组
    """
    import shlex
    
    # 获取 conda 路径（通常在 ~/anaconda3 或 ~/miniconda3）
    conda_base = os.path.expanduser("~/anaconda3")  # 或 ~/miniconda3，根据实际情况修改

    # 通过环境变量传递参数，避免shell转义问题
    env = os.environ.copy()
    env['VIDEO_WEB'] = web
    env['VIDEO_QUERY'] = query

    cmd = (
        f"source {conda_base}/etc/profile.d/conda.sh && "
        f"conda activate video_self_learning && "
        f"cd ../guide && "
        f"python -c \"import os; from youtube import run_get_video; result = run_get_video(os.environ['VIDEO_WEB'], os.environ['VIDEO_QUERY']); print(f'VIDEO_COUNT:{{result}}')\""
    )

    result = subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    # 从输出中提取视频数量
    import re
    match = re.search(r'VIDEO_COUNT:(\d+)', result.stdout)
    video_count = int(match.group(1)) if match else 0
    
    return video_count, result.stdout, result.stderr


# 添加异常处理
def run_auto_convert(web, query, model="gpt-4o"):
    try:
        result = auto_convert(web, query, model=model)
        print(f"Conversion successful for {web} with query: {query} using model: {model}")
        return result
    except Exception as e:
        print(f"Error during conversion for {web} with query: {query}. Error: {e}")
        return None


def run_auto_download(web, query):
    """
    运行 auto_download 并返回下载的视频数量及完整输出

    返回:
    成功时返回 (video_count, stdout, stderr) 元组
    失败时返回 (0, "", error_message)
    """
    try:
        video_count, stdout, stderr = auto_download(web, query)
        print(f"Download successful. Downloaded {video_count} videos.")
        return video_count, stdout, stderr
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error during download. Error: {e}")
        return 0, "", error_msg


def get_converted_results(web, instruction, json_file_path="./evaluation_examples/test_nogdrive_queries_with_videos_with_converted.json"):
    """
    从 JSON 文件中根据 web 和 instruction 查询并返回 planning 和 grounding 结果

    参数:
    web: 应用名称（如 "chrome"）
    instruction: 任务指令
    json_file_path: JSON 文件路径，默认为 test_nogdrive_queries_with_videos_with_converted.json

    返回:
    如果 video_count 为 0，返回 None
    否则返回 (planning_results, grounding_results) 元组
    如果找不到匹配项，返回 None
    """
    try:
        # 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 查找匹配的条目
        for entry in data:
            if entry.get('web') == web and entry.get('instruction') == instruction:
                # 检查视频数量
                video_count = entry.get('video_count', 0)
                if video_count == 0:
                    print(f"Found entry but video_count is 0 for web='{web}', instruction='{instruction}'")
                    return None

                # 提取 planning 和 grounding 结果
                planning_results = entry.get('planning_results', '')
                grounding_results = entry.get('grounding_results', '')

                print(f"Found converted results for web='{web}', instruction='{instruction}'")
                print(f"Video count: {video_count}, Converted video count: {entry.get('converted_video_count', 0)}")

                return planning_results, grounding_results

        # 没有找到匹配项
        print(f"No matching entry found for web='{web}', instruction='{instruction}'")
        return None

    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file. {e}")
        return None
    except Exception as e:
        print(f"Error reading converted results: {e}")
        return None


if __name__ == "__main__":
    # 测试新函数：从已转换的 JSON 文件中获取结果
    print("=" * 80)
    print("Testing get_converted_results function")
    print("=" * 80)

    # 示例：查询已转换的结果
    web = "chrome"
    instruction = "Can you make Bing the main search engine when I look stuff up on the internet?"

    result = get_converted_results(web, instruction)
    if result is not None:
        planning, grounding = result
        print(f"\nPlanning Results (first 500 chars):\n{planning[:500]}...")
        print(f"\nGrounding Results (first 500 chars):\n{grounding[:500] if grounding else 'No grounding results'}...")
    else:
        print("No results found or video_count is 0")

    print("\n" + "=" * 80)
    print("Testing auto_convert function")
    print("=" * 80)

    tasks = [
        # 单应用任务
        ("chrome", "How to find black drip coffee makers on sale between $25 and $60 using Chrome"),
        # ("Calendar", "add event in MacOs Calendar"),
        # ("Finder", "create folder in MacOs Finder"),
        # ("TextEdit", "save document in MacOs TextEdit"),
        # ("Microsoft Word", "insert table in Microsoft Word"),

        # ("Microsoft Excel", "create bar chart with formulas in Microsoft Excel"),
        # ("Microsoft PowerPoint", "create multi-slide presentation in Microsoft PowerPoint"),
        # ("Preview", "annotate and export PDF in MacOs Preview"),
        # ("QuickTime Player", "trim video in MacOs QuickTime Player"),
        # ("Google Chrome", "download and organize file in Google Chrome"),
        # # 跨应用任务
        # ("Google Chrome Microsoft Word", "copy text from Google Chrome to Microsoft Word"),
        # ("Finder TextEdit", "MacOs copy file list from Finder to TextEdit"),
        # ("Calendar Microsoft Word", "MacOs copy event details from Calendar to Microsoft Word"),
        # ("Microsoft Excel Microsoft PowerPoint", "copy chart from Microsoft Excel to Microsoft PowerPoint"),
        # ("Preview TextEdit", "MacOs copy annotation from Preview to TextEdit"),
        # ("Google Chrome Microsoft PowerPoint", "create presentation with research from Google Chrome in Microsoft PowerPoint"),
        # ("Microsoft Excel Microsoft Word", "create data report from Microsoft Excel in Microsoft Word"),
        # ("Finder QuickTime Player", "MacOs organize and play video from Finder in QuickTime Player"),
        # ("Preview Microsoft PowerPoint", "MacOs insert annotated PDF screenshot from Preview in Microsoft PowerPoint"),
        # ("Microsoft Word Microsoft Excel Calendar", "MacOs create meeting materials with Microsoft Excel Microsoft Word and Calendar")
    ]

    for web, query in tasks:
        planning_results, grounding_results = run_auto_convert(web, query)
        # print(f"Results for {web} with query '{query}': {result}")
        print(f"Finished processing {web} with query: {query}")
        print("Planning Results:")
        print(planning_results)
        print("Grounding Results:")
        print(grounding_results)
        print("-" * 80)