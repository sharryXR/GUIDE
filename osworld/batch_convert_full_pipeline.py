#!/usr/bin/env python3
"""
完整的CMD1+2+3批处理脚本
按顺序处理：音频转录+关键帧提取(CMD1) -> OmniParser UI解析(CMD2) -> LLM轨迹生成(CMD3)
"""
import json
import os
from datetime import datetime
from video_self_convert import run_cmd1_only, run_cmd2_only, run_cmd3_only

def load_data():
    json_file = 'evaluation_examples/test_nogdrive_queries_with_videos_with_converted.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    json_file = 'evaluation_examples/test_nogdrive_queries_with_videos_with_converted.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/batch_full_pipeline_{log_time}.log'
    os.makedirs('logs', exist_ok=True)
    
    # 设置内存优化
    os.environ['OMP_NUM_THREADS'] = '8'
    
    data = load_data()
    
    # 筛选需要处理的任务（有视频且cmd3未完成）
    tasks_to_process = []
    for item in data:
        if item.get('video_count', 0) > 0 and not item.get('cmd3_completed', False):
            tasks_to_process.append(item)
    
    total = len(tasks_to_process)
    print(f"{'='*60}")
    print(f"开始完整流程批处理 - 共{total}个任务需要处理")
    print(f"日志文件: {log_file}")
    print(f"{'='*60}\n")
    
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"批处理开始时间: {datetime.now()}\n")
        log.write(f"总任务数: {total}\n")
        log.write(f"{'='*60}\n\n")
        
        for idx, item in enumerate(tasks_to_process, 1):
            web = item['web']
            query = item['query']
            task_id = item['id']
            
            print(f"\n[{idx}/{total}] 处理任务: {task_id}")
            print(f"  Web: {web}")
            print(f"  Query: {query[:80]}...")
            log.write(f"\n{'='*60}\n")
            log.write(f"[{idx}/{total}] 任务ID: {task_id}\n")
            log.write(f"Web: {web}\nQuery: {query}\n")
            log.write(f"开始时间: {datetime.now()}\n")
            
            try:
                # CMD1: 音频转录 + 关键帧提取
                if not item.get('cmd1_completed', False):
                    print(f"  → 执行 CMD1: 音频转录 + 关键帧提取...")
                    log.write(f"\n--- CMD1 开始 ---\n")
                    result1 = run_cmd1_only(web, query)
                    if result1:
                        item['cmd1_completed'] = True
                        save_data(data)
                        print(f"  ✓ CMD1 完成")
                        log.write(f"CMD1 完成: {datetime.now()}\n")
                    else:
                        print(f"  ✗ CMD1 失败")
                        log.write(f"CMD1 失败: {datetime.now()}\n")
                        item['convert_error'] = "CMD1 failed"
                        save_data(data)
                        continue
                else:
                    print(f"  ○ CMD1 已完成，跳过")
                    log.write(f"CMD1 已完成，跳过\n")
                
                # CMD2: OmniParser UI解析
                if not item.get('cmd2_completed', False):
                    print(f"  → 执行 CMD2: OmniParser UI解析...")
                    log.write(f"\n--- CMD2 开始 ---\n")
                    result2 = run_cmd2_only(web, query)
                    if result2:
                        item['cmd2_completed'] = True
                        save_data(data)
                        print(f"  ✓ CMD2 完成")
                        log.write(f"CMD2 完成: {datetime.now()}\n")
                    else:
                        print(f"  ✗ CMD2 失败")
                        log.write(f"CMD2 失败: {datetime.now()}\n")
                        item['convert_error'] = "CMD2 failed"
                        save_data(data)
                        continue
                else:
                    print(f"  ○ CMD2 已完成，跳过")
                    log.write(f"CMD2 已完成，跳过\n")
                
                # CMD3: LLM轨迹生成
                if not item.get('cmd3_completed', False):
                    print(f"  → 执行 CMD3: LLM轨迹生成...")
                    log.write(f"\n--- CMD3 开始 ---\n")
                    result3 = run_cmd3_only(web, query, model="gpt-5.1")
                    # result3返回的是tuple: (planning_results, grounding_results)
                    if result3 and isinstance(result3, tuple) and len(result3) == 2:
                        planning_results, grounding_results = result3
                        if planning_results and grounding_results:
                            item['cmd3_completed'] = True
                            item['planning_results'] = planning_results
                            item['grounding_results'] = grounding_results
                            item['converted_video_count'] = item.get('video_count', 0)
                            
                            # 清除错误标记
                            if 'convert_error' in item:
                                del item['convert_error']
                            
                            save_data(data)
                            print(f"  ✓ CMD3 完成")
                            log.write(f"CMD3 完成: {datetime.now()}\n")
                            log.write(f"Planning结果长度: {len(planning_results)}\n")
                            log.write(f"Grounding结果长度: {len(grounding_results)}\n")
                        else:
                            print(f"  ✗ CMD3 返回空结果")
                            log.write(f"CMD3 返回空结果: {datetime.now()}\n")
                            item['convert_error'] = "CMD3 returned empty planning or grounding results"
                            save_data(data)
                            continue
                    else:
                        print(f"  ✗ CMD3 失败或返回格式错误")
                        log.write(f"CMD3 失败: {datetime.now()}\n")
                        if result3:
                            log.write(f"返回结果类型: {type(result3)}, 值: {result3}\n")
                        item['convert_error'] = "CMD3 failed or returned invalid format"
                        save_data(data)
                        continue
                else:
                    print(f"  ○ CMD3 已完成，跳过")
                    log.write(f"CMD3 已完成，跳过\n")
                
                print(f"  ✓ 任务完成!")
                log.write(f"\n任务完成: {datetime.now()}\n")
                
            except Exception as e:
                print(f"  ✗ 处理失败: {str(e)}")
                log.write(f"\n处理异常: {str(e)}\n")
                item['convert_error'] = f"Exception: {str(e)}"
                save_data(data)
                continue
        
        log.write(f"\n{'='*60}\n")
        log.write(f"批处理结束时间: {datetime.now()}\n")
    
    # 最终统计
    data = load_data()
    completed = sum(1 for item in data if item.get('video_count', 0) > 0 and item.get('cmd3_completed', False))
    total_with_videos = sum(1 for item in data if item.get('video_count', 0) > 0)
    
    print(f"\n{'='*60}")
    print(f"批处理完成!")
    print(f"完成进度: {completed}/{total_with_videos}")
    print(f"日志文件: {log_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
