import json
import os
import time
import logging
import yt_dlp
from typing import Dict, List
import re
import subprocess
from googleapiclient.discovery import build
from fake_useragent import UserAgent
from typing import Optional
import requests
import random
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm3 = ChatOpenAI(
    model="gpt-5-2025-08-07",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

llm_gui = ChatOpenAI(
    model="gpt-5-2025-08-07",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)



# Logging configuration
logging.basicConfig(
    filename="download_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# YouTube Data API key (set via environment variable)
API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")

# Multiple Google Custom Search API configurations (project-level quota rotation)
# Add more (cx, key) pairs from different Google Cloud projects
# Get your API key and CX from: https://programmablesearchengine.google.com/
GOOGLE_SEARCH_CONFIGS = [
    {
        "cx": os.getenv("GOOGLE_CSE_CX", "YOUR_CUSTOM_SEARCH_ENGINE_CX"),
        "key": os.getenv("GOOGLE_CSE_KEY", "YOUR_GOOGLE_API_KEY"),
    },
]

# Legacy single config (for backward compatibility)
cx = GOOGLE_SEARCH_CONFIGS[0]["cx"]
key = GOOGLE_SEARCH_CONFIGS[0]["key"]

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# Initialize spoofed User-Agent
ua = UserAgent()

def search_videos(query, max_results=50):
    """
    Search YouTube videos and return a list of video URLs
    """
    video_urls = []
    next_page_token = None

    while len(video_urls) < max_results:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=min(50, max_results - len(video_urls)),
            pageToken=next_page_token
        )
        response = request.execute()
        print(f"Response: {response}")
        for item in response['items']:
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_urls.append(video_url)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return video_urls

def google_custom_search(query: str, cx: str, key: str, i: int, filter_year: Optional[int] = None):
    print("call search", query, cx, key)
    endpoint = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"q": query, 
              "cx": cx,
              "key": key,
              "filter": 1,
              "start": i,
              "siteSearch": "www.youtube.com",
              "siteSearchFilter": "i"}
    # FIXME this parameter filter_year is not working
    if filter_year is not None:
        params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
    response = requests.get(endpoint, params=params, timeout=30)
    try:
        data = response.json()
    except Exception as e:
        data = {"error": {"message": f"Invalid JSON response: {e}", "code": response.status_code}}
    if response.status_code != 200:
        error_message = data.get("error", {}).get("message", response.text)
        print(f"Google Custom Search API error ({response.status_code}): {error_message}")
        logging.error(f"Google Custom Search API error ({response.status_code}): {error_message}")
    return data
def get_total_duration_from_json(json_data):
    try:
        # Parse JSON data
        data = json.loads(json_data)

        # Iterate through each format entry
        for format_entry in data.get('formats', []):
            # Get fragments list and ensure it is not empty
            fragments = format_entry.get('fragments', [])
            if fragments:
                # Get duration from the first fragment
                total_duration = fragments[0].get('duration', None)
                if total_duration:
                    return total_duration
        return None  # No suitable duration found
    except Exception as e:
        print(f"Error processing JSON: {e}")
        return None

# def save_video_urls(web, op_file, max_results=50):
#     # Base path for saving URL files
#     url_path = f"./urls/{web}"
    
#     # Ensure the directory exists
#     os.makedirs(url_path, exist_ok=True)
    
#     # Read operations from file
#     try:
#         with open(op_file, "r", encoding="utf-8") as file:
#             operations = [line.lstrip("-").strip() for line in file if line.strip()]
#     except:
#         with open(op_file, "r", encoding="GBK") as file:
#             operations = [line.lstrip("-").strip() for line in file if line.strip()]

#     # Process each operation
#     for op in operations:
#         query = f"{web} {op}"

#         url_file = os.path.join(url_path, f"{op}.txt")

#         if os.path.exists(url_file):
#             print(f"Target path {url_file} already exists, skipping download.")
#             continue
        
#         # Search for videos
#         # video_urls = search_videos(query, max_results=max_results)
#         video_urls = []
#         for start_index in range(0, 5):
#             try:
#                 result = google_custom_search(query, cx, key, start_index*10+1)
#                 items = result.get('items', [])
#                 video_urls.extend(item["link"] for item in items)
#             except Exception as e:
#                 print(f"Search error: {e}")
#                 break
        
#         # Save video URLs to a file
#         with open(url_file, "w") as url_file_obj:
#             url_file_obj.writelines(f"{url}\n" for url in video_urls)
        
#         print(f"Video URLs saved to {url_file}")

def save_video_urls_google(web, query, max_results=50):
    """
    Use Google Custom Search API to search for videos (legacy method)
    Requires API key and cx configuration
    """
    # Base path for saving URL files
    url_path = f"./urls/{web}"
    
    # Ensure the directory exists
    os.makedirs(url_path, exist_ok=True)

    # Process each operation
    url_file = os.path.join(url_path, f"{query}.txt")
    
    # If file exists and not empty, skip
    if os.path.exists(url_file) and os.path.getsize(url_file) > 0:
        print(f"Target path {url_file} already exists, skipping download.")
        return
        
    # Search for videos with automatic config rotation on quota exceeded
    video_urls = []
    config_index = 0
    
    for start_index in range(0, 5):
        success = False
        attempts = 0
        
        # Try all available configs if quota exceeded
        while attempts < len(GOOGLE_SEARCH_CONFIGS):
            current_config = GOOGLE_SEARCH_CONFIGS[config_index]
            current_cx = current_config["cx"]
            current_key = current_config["key"]
            
            try:
                result = google_custom_search(query, current_cx, current_key, start_index*10+1)
                
                # Check for quota exceeded error
                if "error" in result:
                    error_code = result.get("error", {}).get("code")
                    error_message = result.get("error", {}).get("message", "Unknown error")
                    
                    if error_code == 429 or "Quota exceeded" in error_message:
                        print(f"⚠️ Config {config_index+1}/{len(GOOGLE_SEARCH_CONFIGS)} quota exceeded, switching to next...")
                        logging.warning(f"Config {config_index+1} quota exceeded: {error_message}")
                        config_index = (config_index + 1) % len(GOOGLE_SEARCH_CONFIGS)
                        attempts += 1
                        continue
                    else:
                        print(f"Search error for query '{query}' (start={start_index*10+1}): {error_message}")
                        logging.error(f"Search error for query '{query}' (start={start_index*10+1}): {error_message}")
                        break
                
                items = result.get('items', [])
                if not items:
                    print(f"No items returned for query '{query}' (start={start_index*10+1}).")
                video_urls.extend(item["link"] for item in items)
                success = True
                break
                
            except Exception as e:
                print(f"Search error: {e}")
                logging.error(f"Search exception: {e}")
                break
        
        # If all configs exhausted, stop searching
        if not success and attempts >= len(GOOGLE_SEARCH_CONFIGS):
            print(f"❌ All {len(GOOGLE_SEARCH_CONFIGS)} API configs exhausted. Please wait for quota reset or add more configs.")
            logging.error(f"All Google Custom Search configs exhausted for query: {query}")
            break
    
    video_urls = list(set(video_urls))  # Deduplicate
    video_urls = video_urls[:max_results]  # Limit count
        
    # Save video URLs to a file
    with open(url_file, "w") as url_file_obj:
        url_file_obj.writelines(f"{url}\n" for url in video_urls)
    
    print(f"💾 Saved {len(video_urls)} video URLs to {url_file}")


def simplify_query_with_llm(llm, query, web):
    """
    Use LLM to generate ONE simplified query variant (max 2 attempts total)
    The simplified query must retain ALL core keywords from original
    """
    prompt = f"""Given a search query for YouTube video tutorials, generate ONE simplified version that is more likely to return results while KEEPING ALL CORE KEYWORDS.

Original query: "{query}"
Context: Searching for {web} tutorials

CRITICAL Requirements:
1. MUST retain ALL core keywords (software names, actions, objects) from the original query
2. ONLY remove filler words like: "how to", "tutorial", "guide", "step by step", "complete", "full", etc.
3. Do NOT change the core meaning or remove important technical terms
4. If the query is already concise, return it unchanged

Return ONLY a JSON array with exactly 2 strings (original first, then simplified):
["original_query", "simplified_query"]

Example 1:
Input: "how to install Python extension in VSCode step by step"
Output: ["how to install Python extension in VSCode step by step", "install Python extension VSCode"]

Example 2:
Input: "create table in LibreOffice Calc"
Output: ["create table in LibreOffice Calc", "create table LibreOffice Calc"]

Now process: "{query}"
"""
    
    try:
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=1.0,
            max_tokens=200,
        )
        response_text = response.content
        
        # Extract JSON array from response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group(0))
            if isinstance(queries, list) and len(queries) >= 1:
                # Ensure we only use max 2 queries
                queries = queries[:2]
                print(f"🤖 LLM generated {len(queries)} query variants:")
                for i, q in enumerate(queries, 1):
                    print(f"   {i}. '{q}'")
                return queries
    except Exception as e:
        print(f"⚠️ LLM query simplification failed: {e}, using original query only")
        logging.warning(f"LLM query simplification failed: {e}")
    
    # Fallback: use only original query if LLM fails
    return [query]


def save_video_urls_ytdlp(web, query, max_results=50, min_results=10):
    """
    Use yt-dlp with ytsearch to search for videos
    No API keys required, uses yt-dlp's built-in YouTube search
    
    Strategy: Try max 2 queries (original + 1 simplified with core keywords retained)
    """
    # Base path for saving URL files
    url_path = f"./urls/{web}"
    
    # Ensure the directory exists
    os.makedirs(url_path, exist_ok=True)

    # Process each operation
    url_file = os.path.join(url_path, f"{query}.txt")
    
    # If file exists and not empty, skip
    if os.path.exists(url_file) and os.path.getsize(url_file) > 0:
        print(f"Target path {url_file} already exists, skipping download.")
        return
        
    # Use LLM to generate max 2 query variants (original + simplified)
    query_variants = simplify_query_with_llm(llm3, query, web)
    print(f"📊 Will try {len(query_variants)} search attempt(s)")
    
    video_urls = []
    
    for idx, search_query in enumerate(query_variants):
        if len(video_urls) >= min_results:
            print(f"✅ Already have {len(video_urls)} videos (>= {min_results}), skipping remaining attempts")
            break
            
        try:
            print(f"\n🔍 Search Attempt {idx+1}/{len(query_variants)}")
            print(f"   Query: '{search_query}'")
            
            # Use yt-dlp's ytsearch feature
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,  # Only extract metadata, don't download
                'force_generic_extractor': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # ytsearch{N}: searches for N results
                ytdlp_query = f"ytsearch{max_results}:{search_query}"
                result = ydl.extract_info(ytdlp_query, download=False)
                
                if result and 'entries' in result:
                    for entry in result['entries']:
                        if entry and 'id' in entry:
                            video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                            if video_url not in video_urls:  # Deduplicate
                                video_urls.append(video_url)
                            
            print(f"   ✅ Found {len(video_urls)} unique videos so far")
            
            # If we got enough results, stop trying simpler queries
            if len(video_urls) >= min_results:
                print(f"✅ Sufficient results ({len(video_urls)} >= {min_results}), stopping search")
                break
            else:
                print(f"⚠️ Only {len(video_urls)} videos found, trying next variant...")
                
        except Exception as e:
            print(f"❌ Search failed for '{search_query}': {e}")
            logging.error(f"yt-dlp search failed for query '{search_query}': {e}")
            continue
    
    # Limit to max_results
    video_urls = video_urls[:max_results]
    
    # Save video URLs to a file
    with open(url_file, "w") as url_file_obj:
        url_file_obj.writelines(f"{url}\n" for url in video_urls)
    
    print(f"💾 Saved {len(video_urls)} video URLs to {url_file}")
    if len(video_urls) < min_results:
        print(f"⚠️ Warning: Only found {len(video_urls)} videos (target was {min_results})")
    
    return len(video_urls)


def save_video_urls(web, query, max_results=50, method='ytdlp'):
    """
    Search and save video URLs using specified method
    
    Args:
        web: Website/application name
        query: Search query
        max_results: Maximum number of results
        method: 'ytdlp' (default) or 'google' (Google Custom Search API)
    """
    if method == 'google':
        return save_video_urls_google(web, query, max_results)
    else:
        return save_video_urls_ytdlp(web, query, max_results)


def download_metadata_using_cmd(video_url, output_path=".", cookies="cookies.txt"):
    """
    Download metadata for a single YouTube video via yt-dlp command
    """
    try:
        # Ensure output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Skip download if metadata already exists
        video_id = video_url.split('=')[-1]
        for file in os.listdir(output_path):
            if file.endswith(".info.json") and f"~~{video_id}." in file:
                print(f"Metadata already exists: {video_url}, skipping download.")
                logging.info(f"Metadata already exists: {video_url}, skipping download.")
                # Extract video ID
                video_id = video_url.split('=')[-1]

                # Find matching metadata file in output directory
                for file in os.listdir(output_path):
                    if file.endswith(".info.json") and f"~~{video_id}." in file:
                        metadata_file_path = os.path.join(output_path, file)
                        break
                else:
                    print(f"Metadata file not found for: {video_url}")
                    logging.warning(f"Metadata file not found for: {video_url}")
                    return False,None

                if os.path.exists(metadata_file_path):
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    # filesize_bytes = metadata.get('filesize') or metadata.get('filesize_approx')
                    # if filesize_bytes:
                    #     filesize_gb = filesize_bytes / (1024 ** 3)  # Convert to GB
                    #     return filesize_gb <= 0.5  # Return whether <= 1GB
                    file_time=metadata.get('duration')
                    title = metadata.get('title')
                    if file_time:
                        return file_time <= 3000,title                  
        # Build yt-dlp command
        metadata_filename = os.path.join(output_path, '%(title)s~~%(id)s')
        command = [
            "yt-dlp",
            video_url,
            "--cookies", cookies,
            # "--cookies-from-browser", "firefox",
            "--ffmpeg-location", "/usr/bin/ffmpeg",
            "-o", metadata_filename,
            "--write-info-json",  # Save metadata to .info.json
            "--skip-download" , # Skip video/audio download
            #"--extractor-args", "youtube:player_client=web"
           ]

        # Debug log
        print(f"Command: {' '.join(command)}")
        logging.info(f"Command: {' '.join(command)}")

        # Run command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check result
        if result.returncode == 0:
            # print(result.stdout)  # Print stdout
            print(f"Metadata download completed: {video_url}")
            logging.info(f"Metadata download succeeded: {video_url}")

            # Extract video ID
            video_id = video_url.split('=')[-1]

            # Find matching metadata file in output directory
            for file in os.listdir(output_path):
                if file.endswith(".info.json") and f"~~{video_id}." in file:
                    metadata_file_path = os.path.join(output_path, file)
                    break
            else:
                print(f"Metadata file not found for: {video_url}")
                logging.warning(f"Metadata file not found for: {video_url}")
                return False,None

            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # filesize_bytes = metadata.get('filesize') or metadata.get('filesize_approx')
                # if filesize_bytes:
                #     filesize_gb = filesize_bytes / (1024 ** 3)  # Convert to GB
                #     return filesize_gb <= 0.5  # Return whether <= 1GB
                file_time=metadata.get('duration')
                title = metadata.get('title')
                if file_time:
                    return file_time <= 3000,title
                
        else:
            # print(result.stderr)  # Print stderr
            print(f"Metadata download failed: {video_url}\nError: {result.stderr}")
            logging.error(f"Metadata download failed: {video_url}\nError: {result.stderr}")

    except Exception as e:
        print(f"Metadata download failed: {video_url}\nError: {e}")
        logging.error(f"Metadata download failed: {video_url}\nError: {e}")
    
    return False,None  # Default to skip on error or missing info

def download_video_using_cmd(video_url, output_path=".", cookies="cookies.txt"):
    """
    Download a single YouTube video via yt-dlp command
    """
    try:
        # Ensure output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Skip download if video already exists
        video_id = video_url.split('=')[-1]
        for file in os.listdir(output_path):
            if file.endswith(".mp4") and f"~~{video_id}." in file:
                print(f"Video already exists: {video_url}, skipping download.")
                logging.info(f"Video already exists: {video_url}, skipping download.")
                return

        # Build yt-dlp command
        command = [
            "yt-dlp",
            video_url,
            "--cookies", cookies,
            # "--cookies-from-browser", "firefox",
            "--ffmpeg-location", "/usr/bin/ffmpeg",
            "-o", os.path.join(output_path, '%(title)s~~%(id)s.%(ext)s'),
            "-f", "bestaudio[ext=m4a]+bestvideo[ext=mp4]/best[ext=mp4]",
            "--merge-output-format", "mp4",
            "--no-overwrites",
            #"--extractor-args", "youtube:player_client=web"
        ]

        # Debug log
        print(f"Command: {' '.join(command)}")
        logging.info(f"Command: {' '.join(command)}")

        # Run command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check result
        if result.returncode == 0:
            # print(result.stdout)  # Print stdout
            print(f"Video download completed: {video_url}")
            logging.info(f"Video download succeeded: {video_url}")
        else:
            # print(result.stderr)  # Print stderr
            print(f"Video download failed: {video_url}\nError: {result.stderr}")
            logging.error(f"Video download failed: {video_url}\nError: {result.stderr}")

    except Exception as e:
        print(f"Download failed: {video_url}\nError: {e}")
        logging.error(f"Download failed: {video_url}\nError: {e}")

def download_audio_using_cmd(video_url, output_path=".", cookies="cookies.txt"):
    """
    Download audio for a single YouTube video via yt-dlp command
    """
    try:
        # Ensure output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Build yt-dlp command
        command = [
            "yt-dlp",
            video_url,
            "--cookies", cookies,
            # "--cookies-from-browser", "firefox",
            "--ffmpeg-location", "/usr/bin/ffmpeg",
            "-o", os.path.join(output_path, '%(title)s~~%(id)s.%(ext)s'),
            "-x",  # Extract audio
            "--audio-format", "mp3",  # Convert to mp3
            "--audio-quality", "0",  # Best quality
            "--no-overwrites",
            #"--extractor-args", "youtube:player_client=web"
        ]

        # Debug log
        print(f"Command: {' '.join(command)}")
        logging.info(f"Command: {' '.join(command)}")

        # Run command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check result
        if result.returncode == 0:
            print(f"Audio download completed: {video_url}")
            logging.info(f"Audio download succeeded: {video_url}")
        else:
            print(f"Audio download failed: {video_url}\nError: {result.stderr}")
            logging.error(f"Audio download failed: {video_url}\nError: {result.stderr}")

    except Exception as e:
        print(f"Audio download failed: {video_url}\nError: {e}")
        logging.error(f"Audio download failed: {video_url}\nError: {e}")

def download_subtitles_using_cmd(video_url, output_path=".", lang_code="en", cookies="cookies.txt"):
    """
    Download subtitles for a single YouTube video via yt-dlp command
    """
    try:
        # Ensure output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Build yt-dlp command
        command = [
            "yt-dlp",
            video_url,
            "--cookies", cookies,
            # "--cookies-from-browser", "firefox",
            "--ffmpeg-location", "/usr/bin/ffmpeg",
            "-o", os.path.join(output_path, '%(title)s~~%(id)s.%(ext)s'),
            "--write-subs",  # Download subtitles
            "--write-auto-subs",  # Download auto-generated subtitles
            "--sub-lang", lang_code,  # Subtitle language
            "--skip-download",  # Subtitles only
            "--no-overwrites",
            #"--extractor-args", "youtube:player_client=web"
        ]

        # Debug log
        print(f"Command: {' '.join(command)}")
        logging.info(f"Command: {' '.join(command)}")

        # Run command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check result
        if result.returncode == 0:
            print(f"Subtitle download completed: {video_url}")
            logging.info(f"Subtitle download succeeded: {video_url}")
        else:
            print(f"Subtitle download failed: {video_url}\nError: {result.stderr}")
            logging.error(f"Subtitle download failed: {video_url}\nError: {result.stderr}")

    except Exception as e:
        print(f"Subtitle download failed: {video_url}\nError: {e}")
        logging.error(f"Subtitle download failed: {video_url}\nError: {e}")


def _clean_subtitle_text(raw_text: str, max_chars: int = 10000) -> str:
    if not raw_text:
        return ""
    text = re.sub(r"\r", "\n", raw_text)
    lines = text.split("\n")
    cleaned_lines = []
    last_line = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if line.lower().startswith("kind:") or line.lower().startswith("language:"):
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"\d{2}:\d{2}:\d{2}\.\d+\s+-->\s+\d{2}:\d{2}:\d{2}\.\d+", line):
            continue
        if re.match(r"\d{2}:\d{2}:\d{2},\d+\s+-->\s+\d{2}:\d{2}:\d{2},\d+", line):
            continue
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line == last_line:
            continue
        cleaned_lines.append(line)
        last_line = line
    cleaned_text = " ".join(cleaned_lines).strip()
    # Split on punctuation to avoid meaningless line breaks
    cleaned_text = re.sub(r"([.!?。！？])\s+", r"\1\n", cleaned_text)
    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars]
    return cleaned_text


def _find_subtitle_file(output_path: str, video_id: str) -> Optional[str]:
    if not os.path.exists(output_path):
        return None
    candidates = []
    for file in os.listdir(output_path):
        if f"~~{video_id}" in file and (file.endswith(".vtt") or file.endswith(".srt")):
            candidates.append(os.path.join(output_path, file))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_subtitle_text(output_path: str, video_id: str) -> str:
    subtitle_file = _find_subtitle_file(output_path, video_id)
    if not subtitle_file:
        return ""
    try:
        with open(subtitle_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        return _clean_subtitle_text(raw_text)
    except Exception as e:
        print(f"Failed to read subtitles: {subtitle_file}\nError: {e}")
        logging.error(f"Failed to read subtitles: {subtitle_file}\nError: {e}")
        return ""


def analyze_gui_and_topic(llm, title: str, subtitle_text: str) -> (bool, str):
    prompt = f"""
    You are a classifier for software/device operation tutorial videos.
    
    Task: Determine if the video shows practical demonstrations of using software, apps, or devices (computer, phone, tablet, etc.).
    
    INCLUDE videos that show:
    - Computer software tutorials (desktop applications, web browsers, etc.)
    - Mobile app tutorials (iOS, Android apps)
    - Device operation guides (phones, tablets, computers)
    - On-screen interface demonstrations
    - Step-by-step GUI operations
    - Any hands-on software/device usage tutorials
    
    EXCLUDE videos that are:
    - Pure theory/concepts without demonstrations
    - Only talking heads or lectures
    - Hardware assembly/repair (not software)
    - Non-tutorial content (reviews, news, entertainment)
    
    If it's a practical demonstration video, extract a clear topic phrase (12-30 words) describing:
    - What software/app/device is being used
    - What task/operation is being performed
    - Key steps or goals
    
    Title: {title}
    Subtitle: {subtitle_text}

    Return strict JSON only with keys: is_gui (true/false), topic (string, empty if not a demonstration video).
    
    Example outputs:
    {{"is_gui": true, "topic": "Installing Python extension in Visual Studio Code editor to enable Python development features"}}
    {{"is_gui": true, "topic": "Creating and formatting a table in LibreOffice Calc spreadsheet application"}}
    {{"is_gui": true, "topic": "Setting up WhatsApp on iPhone mobile device and sending first message"}}
    {{"is_gui": false, "topic": ""}}
    """
    try:
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=1.0,
            max_tokens=500,
        )
        response_text = response.content
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            is_gui = bool(result.get("is_gui", False))
            topic = str(result.get("topic", "") or "").strip()
            return is_gui, topic
    except Exception as e:
        print(f"GUI/topic analysis failed: {e}")
        logging.error(f"GUI/topic analysis failed: {e}")
    return False, ""


def download_video(video_url, output_path=".", cookies="cookies.txt"):
    """
    Download a single YouTube video via yt-dlp
    """
    try:
        # Skip download if video already exists
        if os.path.exists(os.path.join(output_path, f"{video_url.split('=')[-1]}.mp4")):
            print(f"Video already exists: {video_url}, skipping download.")
            logging.info(f"Video already exists: {video_url}, skipping download.")
            return
        # Ensure output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Generate a random User-Agent
        headers = {
            'User-Agent': ua.random
        }

        # Configure yt-dlp options
        ydl_opts = {
            # 'verbose': True,
            'format': "bestvideo+bestaudio/best",  # Best available quality
            # 'merge_output_format': 'mp4',
            'format_sort': ["ext"],
            'outtmpl': os.path.join(output_path, '%(title)s~~%(id)s.%(ext)s'),  # Output filename template
            # 'headers': headers,  # Set request headers
            'noplaylist': True,  # Avoid downloading playlists
            'cookies': cookies  
        }

        # Download with yt-dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Use yt-dlp naming to decide file path
            video_info = ydl.extract_info(video_url, download=False)  # Fetch metadata
            video_filename = ydl.prepare_filename(video_info)  # Resolve filename
            # Skip download if file already exists
            if not os.path.exists(video_filename):  
                print(f"Downloading: {video_url}")
                logging.info(f"Start downloading video: {video_url}")
                ydl.download([video_url])
                print(f"Download completed: {video_url}")
                logging.info(f"Download succeeded: {video_url}")
            else:
                print(f"Video already exists: {video_url}, skipping download.")
                logging.info(f"Video already exists: {video_url}, skipping download.")

    except Exception as e:
        print(f"Download failed: {video_url}\nError: {e}")
        logging.error(f"Download failed: {video_url}\nError: {e}")

def download_videos_from_urls(web, op, url_file, cookies, output_base_path="./videos"):
    """
    Download videos in batch from a URL file, creating a subfolder per operation
    """
    print("web:", web)
    print("op:", op)
    print("url_file:", url_file)
    op_dir = op
    if len(op) > 30:
        op_dir = op[:30]  # Truncate operation name for folder path
    if op_dir.endswith(" "):
        op_dir = op_dir[:-1]
    try:
        # Create output folders for each web operation
        output_path_video = os.path.join("./videos", web, op_dir, "video")
        output_path_audio = os.path.join("./videos", web, op_dir, "audio")
        output_path_subtitle = os.path.join("./videos", web, op_dir, "subtitle")
        
        # Create folder if missing
        if not os.path.exists(output_path_video):
            os.makedirs(output_path_video, exist_ok=True)
        

        # Create folder if missing
        if not os.path.exists(output_path_audio):
            os.makedirs(output_path_audio, exist_ok=True)
        
        # Create folder if missing
        if not os.path.exists(output_path_subtitle):
            os.makedirs(output_path_subtitle, exist_ok=True)
        
        # Read video URLs
        with open(url_file, "r") as file:
            video_urls = [line.strip() for line in file if line.strip()]
        
        if not video_urls:
            print(f"{url_file} has no usable video URLs.")
            return
        
        print(f"Preparing to download {len(video_urls)} videos...")

        i = 0
        # Download each video
        for index, url in enumerate(video_urls, start=1):
            if "www.youtube.com/watch" not in url:
                continue
            if i>=10:
                break
            print(f"Downloading {index}/{len(video_urls)}...")

            download_video_using_cmd(url, output_path_video, cookies=cookies)
            download_audio_using_cmd(url, output_path_audio)
            # download_metadata_using_cmd(url, output_path_meta)
            download_subtitles_using_cmd(url, output_path_subtitle, lang_code="en", cookies=cookies)
            time.sleep(1)  # Wait briefly to reduce rate limiting
            i += 1
    except Exception as e:
        print(f"File read or download failed: {url_file}\nError: {e}")
        logging.error(f"File read or download failed: {url_file}\nError: {e}")

def get_relevance_scores(llm, url_name: Dict[str, str], web: str, op: str) -> Dict[str, float]:
    """
    Calculate relevance scores between video titles and current task using LLM
    
    Parameters:
    llm: LLM client
    url_name: Dictionary of URLs and corresponding video titles
    web: Website name
    op: Current task
    
    Returns:
    url_score: Dictionary of URLs and relevance scores
    """
    url_score = {}
    task_description = f"'{op}' operation on '{web}' "
    
    print(f"Analyzing relevance of {len(url_name)} video titles to task using LLM...")
    
    # Build batch prompt for efficiency
    prompt = f"""
    Analyze the relevance of the following items to the task: "{task_description}".
    Each item may contain a TOPIC and a TITLE.
    IMPORTANT: The TOPIC is more important than the TITLE when judging relevance.
    For each item, provide a relevance score from 0.0 to 1.0, where 1.0 means highly relevant and 0.0 means completely irrelevant.
    Return only scores in JSON format without additional explanation.
    
    Items:
    """
    
    # Add all titles
    for i, (url, title) in enumerate(url_name.items()):
        prompt += f"{i+1}. {title}\n"
    
    prompt += "\nPlease return results in JSON format with index numbers as keys and relevance scores as values, e.g.: {\"1\": 0.8, \"2\": 0.3, ...}"
    print(f"Prompt for LLM: {prompt}")
    try:
        # Call LLM to get relevance scores
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=1.0,
            max_tokens=2000,
        )
        response = response.content
        print(f"LLM response: {response}")
        
        # Parse JSON result from LLM response
        # Find JSON part in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            scores_dict = json.loads(json_match.group(0))
            
            # Map indices back to URLs
            for idx, (url, _) in enumerate(url_name.items(), 1):
                str_idx = str(idx)
                if str_idx in scores_dict:
                    url_score[url] = float(scores_dict[str_idx])
                    print(f"Idx:{str_idx}, URL: {url}, Relevance Score: {url_score[url]}")
                else:
                    print(f"Warning: Relevance score not found for title index {idx}")
                    url_score[url] = 0.0
        else:
            print("Warning: Could not parse JSON result from LLM response")
            # If parsing fails, request for each URL individually
            for url, title in url_name.items():
                single_prompt = f"""
                Analyze the relevance of video title "{title}" to the task: "{task_description}".
                Provide a relevance score from 0.0 to 1.0, where 1.0 means highly relevant and 0.0 means completely irrelevant.
                Return only a number without any explanation.
                """
                try:
                    score_response = llm.invoke(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": single_prompt},
                                ],
                            },
                        ],
                        temperature=0.5,
                        max_tokens=50,
                    )
                    # Extract number
                    score_match = re.search(r'[0-9]*\.?[0-9]+', score_response.content)
                    if score_match:
                        url_score[url] = float(score_match.group(0))
                    else:
                        print(f"Warning: Could not parse relevance score from LLM response, URL: {url}")
                        url_score[url] = 0.0
                except Exception as e:
                    print(f"Error calculating relevance for individual URL: {e}")
                    url_score[url] = 0.0
    except Exception as e:
        print(f"Error calculating relevance scores: {e}")
        # Assign default scores when errors occur
        for url in url_name:
            url_score[url] = 0.0
    
    print(f"Relevance analysis complete, processed {len(url_score)} URLs")
    return url_score


def get_relevance_scores_from_titles_and_subtitles(
    llm_relevance,
    llm_gui_agent,
    url_name: Dict[str, str],
    web: str,
    op: str,
    subtitle_output_path: str,
    cookies: str,
    lang_code: str = "en"
) -> Dict[str, float]:
    """
    Download subtitles, classify GUI-related videos, extract topic, then compute relevance
    using title + topic against the task.
    """
    url_title_for_relevance = {}
    for url, title in url_name.items():
        try:
            download_subtitles_using_cmd(url, subtitle_output_path, lang_code=lang_code, cookies=cookies)
            video_id = url.split("=")[-1]
            subtitle_text = _load_subtitle_text(subtitle_output_path, video_id)
            is_gui, topic = analyze_gui_and_topic(llm_gui_agent, title, subtitle_text)
            print(f"Video: {title}\nIs GUI: {is_gui}\nTopic: {topic}\n")
            # input("Press Enter to continue...")
            if not is_gui:
                print(f"Non-GUI video, skipping: {title}")
                continue
            if topic:
                combined_title = f"TOPIC (higher priority): {topic}. TITLE: {title}. TOPIC: {topic}"
            else:
                combined_title = f"TITLE: {title}"
            url_title_for_relevance[url] = combined_title
        except Exception as e:
            print(f"Subtitle/topic processing failed: {url}\nError: {e}")
            logging.error(f"Subtitle/topic processing failed: {url}\nError: {e}")
            continue

    if not url_title_for_relevance:
        print("No GUI-related videos available for relevance scoring.")
        return {}

    return get_relevance_scores(llm_relevance, url_title_for_relevance, web, op)



# Filter URLs using video length and titles with the LLM
def select_video_urls(llm,web,op,url_file_path, output_path,cookies,max_results=3):
    """
    Filter video URLs from a file and return those that meet criteria
    """
    url_selected_file_path = url_file_path.replace(".txt", "_selected.txt")
    if os.path.exists(url_selected_file_path) and os.path.getsize(url_selected_file_path) > 0:
        print(f"Target path {url_selected_file_path} already exists and not empty, skipping filtering.")
        return
    selected_urls = []
    url_name =dict() 
    # Truncate op for folder name
    op_dir = op
    if len(op) > 30:
        op_dir = op[:30]
    if op_dir.endswith(" "):
        op_dir = op_dir[:-1]
    output_path_meta = os.path.join("./videos", web, op_dir, "meta")
    output_path_subtitle = os.path.join("./videos", web, op_dir, "subtitle_select")
    try:
        with open(url_file_path, "r") as file:
            video_urls = [line.strip() for line in file if line.strip()]
        
        if not video_urls:
            print(f"{url_file_path} has no usable video URLs.")
            return 
        i = 0
        # Iterate through each video URL
        for url in video_urls:
            if "www.youtube.com/watch" not in url:
                continue
            # Download metadata first and check duration
            should_download,video_title = download_metadata_using_cmd(url, output_path_meta, cookies=cookies)
            if not should_download:
                print(f"Video duration exceeds 3000s or metadata read failed, skipping: {url}")
                continue
            # Filter titles with invalid filename characters
            if ':' in video_title or '|' in video_title or'\\' in video_title or '/' in video_title or '?'in video_title or '*' in video_title or '"' in video_title or '<' in video_title or '>' in video_title:
                print(f"Title contains invalid characters, skipping: {video_title}")
                continue
            url_name[url] = video_title
            i+=1
            if i >= (5*max_results):
                break
        # Check if there are enough videos
        if not url_name:
            print(f"No video URLs meet the criteria.")
            return
            
        # Use relevance to sort URLs based on titles + topics from subtitles
        relevance_scores = get_relevance_scores_from_titles_and_subtitles(
            llm,
            llm_gui,
            url_name,
            web,
            op,
            output_path_subtitle,
            cookies,
            lang_code="en"
        )
        
        # Sort by relevance score
        sorted_urls = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top max_results URLs
        # selected_urls = [url for url, score in sorted_urls[:max_results]]
        # For items beyond rank 1, drop those with relevance < 0.5
        selected_urls = []
        if len(sorted_urls)>=1:
            selected_urls.append(sorted_urls[0][0])  # Always keep the top URL
        if len(sorted_urls) > 1:
            for url, score in sorted_urls[1:max_results]:
                if score >= 0.5:
                    selected_urls.append(url)

        # Write selected URLs to file
        with open(url_selected_file_path, "w") as file:
            for url in selected_urls:
                file.write(f"{url}\n")
                
        print(f"Selected {len(selected_urls)} relevant video URLs and saved to {url_selected_file_path}")
        
    except Exception as e:
        print(f"Error filtering video URLs: {e}")
        
    return selected_urls
        

## External entry function
def run_get_video(web, query):
    # Step 1: Save video URLs
    # print(f"Saving video URLs for web: {web}, query: {query}")
    # save_video_urls(web, query, max_results=50)
    # # Step 2: Filter video URLs
    # url_path = f"./urls/{web}"
    # url_file_path = os.path.join(url_path, f"{query}.txt")
    # output_path = os.path.join("./videos", web, query)
    cookies1 = "cookies.txt"
    cookies2 = "cookies2.txt"
    cookies3 = "cookies3.txt"
    cookies4 = "cookies4.txt"
    cookies = random.choice([cookies1, cookies2, cookies3, cookies4])
    # selected_urls = select_video_urls(llm3,web,query,url_file_path, output_path,cookies,max_results=2)
    # print(f"Selected URLs: {selected_urls}")

    # Step 3: Download videos
    # Get all URL files
    i = 0
    url_path = f"./urls/{web}"
    url_file_path = os.path.join(url_path, f"{query}_selected.txt")

    selected_url_count = 0
    if os.path.exists(url_file_path):
        with open(url_file_path, "r") as file:
            selected_url_count = len([line for line in file if line.strip()])
    print(f"Selected URL count: {selected_url_count}")

    download_videos_from_urls(web, query, url_file_path, cookies)
    return selected_url_count
    # time.sleep(60)

    

if __name__ == "__main__":
    web="Vscode"
    query="Vscode install Python extension"
    run_get_video(web, query)

