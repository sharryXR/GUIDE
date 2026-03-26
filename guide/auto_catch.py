from guide.youtube import run_get_video
from guide.asr import run_asr
from guide.keyframe_subtitle import run_keyframe_subtitle
from guide.run_sumvideo import run_sumvideo
import argparse
def run_auto_catch(web, query):
    # run_get_video(web, query)
    run_asr(web, query)
    run_sumvideo(web, query)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Auto Catch with specified web and query.")
    parser.add_argument('--web', type=str, required=True, help='The web application to use.')
    parser.add_argument('--query', type=str, required=True, help='The query to process.')
    args = parser.parse_args()
    
    run_auto_catch(args.web, args.query)

# if __name__ == '__main__':
#     tasks = [
#         # 单应用任务
#         # ("Google Chrome", "add bookmark in Google Chrome"),
#         ("Calendar", "add event in MacOs Calendar"),
#         # ("Finder", "create folder in MacOs Finder"),
#         # ("TextEdit", "save document in MacOs TextEdit"),
#         # ("Microsoft Word", "insert table in Microsoft Word"),
#         # ("Microsoft Excel", "create bar chart with formulas in Microsoft Excel"),
#         # ("Microsoft PowerPoint", "create multi-slide presentation in Microsoft PowerPoint"),
#         # ("Preview", "annotate and export PDF in MacOs Preview"),
#         # ("QuickTime Player", "trim video in MacOs QuickTime Player"),
#         # ("Google Chrome", "download and organize file in Google Chrome"),
#         # # 跨应用任务
#         # ("Google Chrome Microsoft Word", "copy text from Google Chrome to Microsoft Word"),
#         # ("Finder TextEdit", "MacOs copy file list from Finder to TextEdit"),
#         # ("Calendar Microsoft Word", "MacOs copy event details from Calendar to Microsoft Word"),
#         # ("Microsoft Excel Microsoft PowerPoint", "copy chart from Microsoft Excel to Microsoft PowerPoint"),
#         # ("Preview TextEdit", "MacOs copy annotation from Preview to TextEdit"),
#         # ("Google Chrome Microsoft PowerPoint", "create presentation with research from Google Chrome in Microsoft PowerPoint"),
#         # ("Microsoft Excel Microsoft Word", "create data report from Microsoft Excel in Microsoft Word"),
#         # ("Finder QuickTime Player", "MacOs organize and play video from Finder in QuickTime Player"),
#         # ("Preview Microsoft PowerPoint", "MacOs insert annotated PDF screenshot from Preview in Microsoft PowerPoint"),
#         # ("Microsoft Word Microsoft Excel Calendar", "MacOs create meeting materials with Microsoft Excel Microsoft Word and Calendar")
#     ]

#     for web, query in tasks:
#         run_auto_catch(web, query)