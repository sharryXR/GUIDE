import subprocess
import os

def auto_convert(web, query):
    """
    自动转换函数，接收 web 和 query 参数。
    """
    print(f"Converting {web} with query: {query}")
        
    import shlex
    web_escaped = shlex.quote(web)
    query_escaped = shlex.quote(query)
    # 转为命令行参数
    args = f'--web {web_escaped} --query {query_escaped}'
    
    # 获取 conda 路径（通常在 ~/anaconda3 或 ~/miniconda3）
    conda_base = os.path.expanduser("~/anaconda3")  # 或 ~/miniconda3，根据实际情况修改
    
    # 脚本1：auto_catch.py 运行在 video_self_learning 环境
    cmd1 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate video_self_learning && ' \
           f'cd . && ' \
           f'python auto_catch.py {args}'
    
    # 脚本2：auto_omni.py 运行在 omni 环境
    cmd2 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate omni && ' \
           f'cd ./OmniParser && ' \
           f'python auto_omni.py {args}'
    
    # 脚本3：action_annotation4.py 进行动作标注
    cmd3 = f'source {conda_base}/etc/profile.d/conda.sh && ' \
           f'conda activate video_self_learning && ' \
           f'cd . && ' \
           f'python -m guide.action_annotation {args}'
    
    # 组合为完整命令（使用 bash 执行）
    full_cmd = f'{cmd1} && {cmd2} && {cmd3}'
    
    # 运行（使用 bash -c 执行，并设置 executable）
    subprocess.run(full_cmd, shell=True, executable='/bin/bash')
    
    # 读取最终的标注结果并返回
    output_dir = f"./videos/{web}/{query}/Labeled/devided"
    grounding_dir = os.path.join(output_dir, "grounding")
    planning_dir = os.path.join(output_dir, "planning")

    grounding_results = ""
    planning_results = ""
    if not os.path.exists(grounding_dir):
        print(f"Warning: Grounding directory does not exist: {grounding_dir}")
    else:
        for filename in os.listdir(grounding_dir):
            i=0
            if filename.endswith('_grounding.txt'):   
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


# 添加异常处理
def run_auto_convert(web, query):
    try:
        result = auto_convert(web, query)
        print(f"Conversion successful for {web} with query: {query}")
        return result
    except Exception as e:
        print(f"Error during conversion for {web} with query: {query}. Error: {e}")
        return None


if __name__ == "__main__":
    tasks = [
        # 单应用任务
        # ("Google Chrome", "add bookmark in Google Chrome"),
        # ("Calendar", "add event in MacOs Calendar"),
        # ("Finder", "create folder in MacOs Finder"),
        # ("TextEdit", "save document in MacOs TextEdit"),
        # ("Microsoft Word", "insert table in Microsoft Word"),
        # ("Microsoft Excel", "create bar chart with formulas in Microsoft Excel"),
        ("chrome","How to find drip coffee makers on sale between $25-60 with a black finish in Chrome browser"),
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