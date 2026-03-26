---
language:
  - en
license: apache-2.0
task_categories:
  - video-classification
  - visual-question-answering
tags:
  - gui-agent
  - video-retrieval
  - action-annotation
  - osworld
  - tutorial-videos
  - domain-bias
  - planning
  - grounding
size_categories:
  - 100<n<1K
---

# GUIDE Dataset

Dataset for the paper **"GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation"**.

## Overview

GUIDE (GUI Unbiasing via Instructional-Video Driven Expertise) is a training-free framework that resolves domain bias in GUI agents by retrieving tutorial videos from YouTube and automatically generating domain-specific Planning and Grounding knowledge. This dataset contains the complete video corpus, automated annotations, and pre-computed knowledge for reproducing all experiments on the [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark.

## Dataset Contents

### 1. Tutorial Videos (`videos/`, ~21 GB)

~427 YouTube tutorial videos covering 10 desktop application domains, retrieved via GUIDE's subtitle-driven Video-RAG pipeline. Each video directory contains:

| Subdirectory | Description |
|-------------|-------------|
| `video/` | Original MP4 video file |
| `meta/` | yt-dlp metadata JSON (title, duration, upload date, formats) |
| `subtitle/` | Video subtitle files |
| `audio/` | Extracted audio (MP3) |
| `audios_text/` | ASR transcription from OpenAI Whisper (base model, word-level timestamps) |
| `keyframes_*/` | Extracted keyframes (uniform sampling + MOG2 background subtraction) |
| `OmniParser_Pic/` | UI element detection results from OmniParser (bounding boxes, element types, text labels) |
| `Labeled_gpt-4.1/` | Action annotations from VLM inverse dynamics inference |
| `Labeled_gpt-4.1/consolidated/` | Consolidated per-video annotations (Thought & Action NLP) |
| `Labeled_gpt-4.1/divided/planning/` | Extracted Planning knowledge per video |
| `Labeled_gpt-4.1/divided/grounding/` | Extracted Grounding knowledge per video |

### 2. Video URLs (`urls/`, ~3 MB)

YouTube URL lists organized into 70 directories by application/query combination. These can be used to re-download videos or to understand the retrieval scope.

### 3. Pre-computed Results (`converted_results/`, ~4.5 MB)

`test_nogdrive_queries_with_videos_with_converted.json` contains pre-computed Planning and Grounding knowledge for all **361 OSWorld evaluation tasks**, ready for direct injection into GUI agents without re-running the annotation pipeline.

**Entry format:**
```json
{
  "id": "bb5e4c0d-f964-439c-97b6-bdb9747de3f4",
  "web": "chrome",
  "instruction": "Set Bing as the default search engine in Google Chrome",
  "query": "How to set Bing as the default search engine in Google Chrome",
  "video_count": 2,
  "converted_video_count": 2,
  "planning_results": "The planning trajectory of Demo 1: ...\nThe planning trajectory of Demo 2: ...",
  "grounding_results": "The grounding trajectory of Demo 1: ...\nThe grounding trajectory of Demo 2: ...",
  "cmd1_completed": true,
  "cmd2_completed": true,
  "cmd3_completed": true
}
```

**Field descriptions:**
- `id`: OSWorld task UUID (matches `evaluation_examples/examples/{domain}/{id}.json`)
- `web`: Application domain (chrome, gimp, libreoffice_calc, etc.)
- `instruction`: Original OSWorld task instruction
- `query`: GUIDE-generated search query for YouTube video retrieval
- `video_count`: Number of videos retrieved by Video-RAG
- `converted_video_count`: Number of videos successfully annotated
- `planning_results`: Concatenated Planning knowledge from all annotated videos -- contains execution workflows, step sequences, and key considerations (coordinate-free)
- `grounding_results`: Concatenated Grounding knowledge -- UI element catalog with visual descriptions (color, shape, text labels), screen-relative positions, and inferred functions
- `cmd1_completed`: ASR + keyframe extraction stage completed
- `cmd2_completed`: OmniParser UI element detection completed
- `cmd3_completed`: VLM action annotation + knowledge decomposition completed

### 4. Verification Report (`video_verification_report.json`)

Data integrity report: 361 total tasks, 298 matched with downloaded videos, coverage statistics per domain.

## Statistics

| Metric | Value |
|--------|-------|
| Total OSWorld tasks | 361 |
| Tasks with retrieved videos | 299 (82.8%) |
| Tasks with 2+ videos | 42.7% |
| Total annotated videos | ~427 |
| Application domains | 10 |
| Total video data size | ~21 GB |
| Annotation model | GPT-5.1 (default) |
| Annotation cost (GPT-5.1) | ~$0.25/video, ~$115 total |

### Per-Domain Task Distribution

| Domain | Abbrev. | Tasks | Domain | Abbrev. | Tasks |
|--------|---------|-------|--------|---------|-------|
| Google Chrome | Chrome | 46 | Ubuntu OS | OS | 24 |
| GIMP | GIMP | 26 | Thunderbird | ThBrd | 15 |
| LibreOffice Calc | Calc | 47 | VLC Media Player | VLC | 17 |
| LibreOffice Impress | Impress | 45 | VS Code | VSCode | 23 |
| LibreOffice Writer | Writer | 23 | Cross-application | Multi | 93 |

## Usage

### Loading Pre-computed Results (Recommended)

```python
import json

with open("converted_results/test_nogdrive_queries_with_videos_with_converted.json") as f:
    tasks = json.load(f)  # 361 entries

# Example: Get knowledge for a specific task
task = tasks[0]
planning = task["planning_results"]   # Inject into agent's planning context
grounding = task["grounding_results"] # Inject into agent's grounding context
```

### Re-running the Annotation Pipeline

To regenerate annotations from scratch using the GUIDE pipeline:

```bash
# Clone the GUIDE repository
git clone https://github.com/sharryXR/GUIDE.git
cd GUIDE

# Run batch conversion for all tasks
cd osworld
python batch_convert_full_pipeline.py
```

See the [GUIDE repository](https://github.com/sharryXR/GUIDE) for full documentation.

## Annotation Pipeline Details

The annotations were generated by GUIDE's three-stage pipeline:

1. **Video-RAG Retrieval**: Subtitle-driven 3-stage filtering (domain classification -> topic extraction -> relevance matching) from YouTube, selecting top-K (K <= 2) videos per task.

2. **Inverse Dynamics Annotation**: For each video:
   - ASR via OpenAI Whisper (base model, word-level timestamps)
   - Keyframe extraction with MOG2 background subtraction
   - UI element parsing via OmniParser (bounding boxes, element types)
   - VLM inference on consecutive keyframe pairs to produce Thought & Action annotations
   - Meaningful filter removes >91% of non-GUI/idle frames

3. **Knowledge Decomposition**: Annotations decomposed into:
   - **Planning**: Coordinate-free execution workflows and key considerations
   - **Grounding**: Up to 15 key UI elements with visual descriptions and inferred functions

## Citation

The paper is currently under anonymous review. The arXiv preprint and full citation will be available soon.

```bibtex
@article{guide2026,
  title={{GUIDE}: Resolving Domain Bias in {GUI} Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This dataset is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The tutorial videos are sourced from YouTube and are subject to their original creators' terms. This dataset is provided for academic research purposes.
