# GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation

<p align="center">
  <b>G</b>UI <b>U</b>nbiasing via <b>I</b>nstructional-video <b>D</b>riven <b>E</b>xpertise
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#pipeline-overview">Pipeline</a> |
  <a href="#evaluation-on-osworld">Evaluation</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#citation">Citation</a>
</p>

---

## Abstract

Large vision-language models have endowed GUI agents with strong general capabilities for interface understanding and interaction. However, due to insufficient exposure to domain-specific software operation data during training, these agents exhibit significant **domain bias** -- they lack familiarity with the specific operation workflows (**planning**) and UI element layouts (**grounding**) of particular applications, limiting their real-world task performance.

**GUIDE** is a training-free, plug-and-play framework that resolves GUI agent domain bias by autonomously acquiring domain-specific expertise from web tutorial videos through a retrieval-augmented automated annotation pipeline. The framework delivers consistent **4.5--7.5 percentage-point improvements** across three distinct agent architectures on the [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark (361 tasks, 10 application domains) without any model fine-tuning.

### Main Results

| Agent | Type | Baseline | + Planning | + Planning & Grounding | Improvement |
|-------|------|----------|-----------|----------------------|-------------|
| **Seed-1.8** | Single-model (closed) | 37.14% | 43.93% | **44.62%** | **+7.48pp** |
| **Qwen3-VL-8B** | Single-model (open, 8B) | 33.90% | 38.93% | **39.73%** | **+5.83pp** |
| **AgentS3** | Multi-agent (GPT-5.2 + Seed-1.8) | 50.18% | -- | **54.65%** | **+4.47pp** |

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- FFmpeg (`sudo apt install ffmpeg`)
- [OmniParser](https://github.com/microsoft/OmniParser) (for UI element detection in annotation stage)
- Docker (for OSWorld evaluation VM environment)

### Step 1: Clone and Set Up GUIDE Pipeline

```bash
git clone https://github.com/sharryXR/GUIDE.git
cd GUIDE

conda create -n guide python=3.10 -y
conda activate guide

# Install GUIDE pipeline dependencies
pip install -r requirements.txt
```

### Step 2: Set Up OSWorld Evaluation Environment

```bash
cd osworld
pip install -e .
cd ..
```

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys (see table below)
```

### Required API Keys

| Variable | Service | Used For |
|----------|---------|----------|
| `OPENAI_API_KEY` | OpenAI | GPT-4.1 / GPT-5.x annotation, AgentS3 Worker |
| `DOUBAO_API_KEY` | ByteDance Doubao | Seed-1.8 agent (grounding, single-agent experiments) |
| `OPENAI_QWEN_API_KEY` | Alibaba DashScope | Qwen2.5-VL / Qwen3-VL-8B models |
| `YOUTUBE_API_KEY` | Google | YouTube Data API v3 for video search |
| `GOOGLE_CSE_KEY` / `GOOGLE_CSE_CX` | Google | Custom Search Engine (video search fallback) |
| `OSWORLD_BASE_URL` | Self-hosted | OSWorld Docker server address |
| `OSWORLD_TOKEN` | Self-hosted | OSWorld Docker server auth token |
| `AZURE_OPENAI_API_KEY` (optional) | Azure | Azure OpenAI endpoint |

---

## Quick Start

### End-to-End Pipeline

Given a task description, GUIDE retrieves tutorial videos from YouTube, extracts knowledge, and generates dual-channel Planning + Grounding information:

```python
from guide.auto_work import run_auto_convert

web = "chrome"
query = "How to set Bing as the default search engine in Google Chrome"

planning_results, grounding_results = run_auto_convert(web, query)
```

### Step-by-Step Usage

```python
# Stage 1: Video-RAG retrieval
from guide.youtube import run_get_video
run_get_video(web="chrome", query="How to clear browsing history in Chrome")

# Stage 2a: ASR transcription (Whisper)
from guide.asr import run_asr
run_asr(web="chrome", query="How to clear browsing history in Chrome")

# Stage 2b: Keyframe extraction (uniform sampling + MOG2 background subtraction)
from guide.run_sumvideo import run_sumvideo
run_sumvideo(web="chrome", query="How to clear browsing history in Chrome")

# Stage 2c: UI element parsing (OmniParser) - requires separate environment
# Stage 2d: VLM action annotation (inverse dynamics)
# python -m guide.action_annotation --web chrome --query "..."
```

---

## Pipeline Overview

GUIDE operates through three stages:

### Stage 1: Subtitle-Driven Video-RAG

A three-step progressive filtering pipeline retrieves relevant tutorial videos from YouTube:

1. **Domain Classification** -- LLM analyzes video title + subtitles to filter non-GUI content (94.3% accuracy, 100% precision for GUI domain classification)
2. **Topic Extraction** -- Distills 12--30 word semantic descriptors from subtitles (more reliable than titles alone; 77.3% achieve perfect 1.0 relevance)
3. **Dual-Anchored Relevance Matching** -- Scores candidates on 0.0--1.0 scale with adaptive top-K selection (K <= 2)

On the 361-task OSWorld benchmark, **82.8%** of tasks retrieved at least one relevant video, with **42.7%** retrieving a second video for multi-perspective references. Total: **~427 selected videos**.

### Stage 2: Automated Annotation (Inverse Dynamics Model)

Each retrieved video is processed through a fully automated pipeline:

1. **ASR** -- OpenAI Whisper (base model) produces word-level timestamped subtitles
2. **Keyframe Extraction** -- Uniformly-spaced frames with subtitle-aligned MOG2 background subtraction to identify meaningful changes (foreground pixel threshold > 10,000)
3. **UI Element Parsing** -- [OmniParser](https://github.com/microsoft/OmniParser) detects interactive elements (buttons, menus, text fields) with bounding boxes and structured JSON descriptions
4. **VLM Action Inference** -- Analyzes consecutive keyframe pairs (s_t, s_{t+1}) with UI element graphs (E_t, E_{t+1}), video topic (T_topic), and subtitle context (C_sub) using an inverse dynamics paradigm

The **Meaningful** filter correctly removes **>91%** of non-GUI frames and idle no-action frames.

Default annotation model: **GPT-5.1** ($0.25/video). Alternatives: GPT-4.1-Mini ($0.048), Seed-1.8 ($0.029), Qwen3-VL-8B ($0.014). Total benchmark cost with GPT-5.1: **~$115** for 427 videos.

### Stage 3: Knowledge Decomposition

Annotated trajectories are decomposed into two complementary channels:

- **Planning Knowledge** -- Execution workflows, step sequences, key considerations, and decision points. Deliberately coordinate-free for cross-resolution transferability. Contributes **~85--91%** of total improvement.
- **Grounding Knowledge** -- UI element catalog (up to 15 key elements) with visual descriptions (color, shape, text labels), screen-relative position, and inferred function. Provides complementary **+0.69--0.80pp** gains, strongest in complex UI domains (GIMP, Calc).

### Knowledge Injection

GUIDE supports two integration modes:

- **Mode A (Multi-Agent)**: Planning knowledge injected into AgentS3's Worker system prompt; Grounding knowledge supplied to Grounding Agent per action query.
- **Mode B (Single-Model)**: Both channels injected into a unified system prompt with structured Chain-of-Thought template that enforces active knowledge referencing.

Both modes include graceful degradation and verification-first instructions -- agents always prioritize their own screenshot observations when conflicts arise.

---

## Evaluation on OSWorld

### Benchmark

[OSWorld](https://github.com/xlang-ai/OSWorld) (NeurIPS 2024) contains 369 real-world desktop tasks executed inside live Ubuntu virtual machines at 1920x1080 resolution. We evaluate on **361 tasks** (excluding 8 Google Drive-dependent tasks) spanning **10 application domains**:

| Domain | Tasks | Domain | Tasks |
|--------|-------|--------|-------|
| Chrome | 46 | OS (Ubuntu) | 24 |
| GIMP | 26 | Thunderbird | 15 |
| LibreOffice Calc | 47 | VLC | 17 |
| LibreOffice Impress | 45 | VS Code | 23 |
| LibreOffice Writer | 23 | Multi-app | 93 |

### Running Experiments

**Prerequisites**: Set up OSWorld Docker environment following the [OSWorld documentation](https://github.com/xlang-ai/OSWorld), then configure `OSWORLD_BASE_URL` and `OSWORLD_TOKEN` in `.env`.

```bash
cd osworld

# AgentS3 + GUIDE (multi-agent: GPT-5.2 Worker + Seed-1.8 Grounding)
python run_local_withvideo.py

# Seed-1.8 + GUIDE (single-agent, full dual-channel)
python run_multienv_seed1_8.py

# Qwen3-VL-8B + GUIDE (single-agent, full dual-channel)
python run_multienv_qwen3vl_full_annotation.py

# Qwen3-VL-8B baseline (no GUIDE knowledge)
python run_multienv_qwen3vl_no_annotation.py

# View results
python show_result.py --result_dir results/
```

### Detailed Per-Domain Results

**Seed-1.8 + GUIDE (Planning & Grounding)**:

| | Chrome | GIMP | Calc | Impress | Writer | OS | ThBrd | VLC | VSCode | Multi | **Overall** |
|-|--------|------|------|---------|--------|-----|-------|-----|--------|-------|-------------|
| Baseline | 36.87 | 26.92 | 29.79 | 43.09 | 34.77 | 45.83 | 66.67 | 47.06 | 60.87 | 26.88 | **37.14** |
| +GUIDE | 47.74 | 42.31 | 48.94 | 45.31 | 56.51 | 50.00 | 73.33 | 52.32 | 65.22 | 25.74 | **44.62** |

**AgentS3 + GUIDE (Planning & Grounding)**:

| | Chrome | GIMP | Calc | Impress | Writer | OS | ThBrd | VLC | VSCode | Multi | **Overall** |
|-|--------|------|------|---------|--------|-----|-------|-----|--------|-------|-------------|
| Baseline | 41.18 | 38.46 | 51.06 | 44.62 | 52.17 | 70.83 | 73.33 | 73.91 | 73.91 | 40.32 | **50.18** |
| +GUIDE | 49.85 | 53.85 | 65.96 | 46.88 | 65.22 | 70.83 | 80.00 | 56.25 | 82.61 | 37.10 | **54.65** |

### Key Evaluation Files

| File | Role |
|------|------|
| `osworld/mm_agents/seed_agent.py` | Seed-1.8 agent with GUIDE dual-channel prompt injection (Mode B) |
| `osworld/mm_agents/qwen3vl_agent.py` | Qwen3-VL-8B agent with thinking mode + GUIDE injection (Mode B) |
| `osworld/mm_agents/prompts.py` | Prompt templates for single-model agents |
| `osworld/mm_agents/agent.py` | Base PromptAgent class |
| `osworld/new_gui_agents_with_video/s3/agents/worker.py` | AgentS3 Worker (receives Planning knowledge, Mode A) |
| `osworld/new_gui_agents_with_video/s3/agents/grounding.py` | AgentS3 Grounding Agent (receives Grounding knowledge, Mode A) |
| `osworld/new_gui_agents_with_video/s3/memory/procedural_memory.py` | All agent prompt templates and memory |
| `osworld/video_self_convert.py` | Video annotation pipeline coordinator |

---

## Dataset

The complete GUIDE dataset is available on HuggingFace:

**[GUIDE-dataset on HuggingFace](https://huggingface.co/datasets/sharryXR/GUIDE-dataset)**

### Contents

| Component | Description | Size |
|-----------|-------------|------|
| `videos/` | 299 annotated video directories (453 MP4 files) across 10 application domains. Each video includes: MP4 file, yt-dlp metadata JSON, subtitles, extracted audio, ASR transcription, keyframes, OmniParser UI element annotations, and action annotations. The default annotation directory `Labeled_gpt-5.1/` contains **GPT-5.1** annotations; some videos also include ablation annotations from Qwen3-VL-8B (50), GPT-4.1-Mini (50), and Seed-1.8 (33). | ~21 GB |
| `urls/` | YouTube URL lists for all 70 app/query combinations used in retrieval. | ~3 MB |
| `converted_results/` | Pre-computed Planning and Grounding knowledge for all 361 OSWorld tasks, ready for direct injection into agents. | ~4.5 MB |
| `video_verification_report.json` | Data integrity verification report (361 tasks, 298 matched, coverage statistics). | <1 MB |

### Using Pre-computed Results

To reproduce experiments without re-running the full annotation pipeline, use the pre-computed `converted_results/` JSON:

```python
import json

with open("converted_results/test_nogdrive_queries_with_videos_with_converted.json") as f:
    tasks = json.load(f)  # List of 361 task entries

for task in tasks:
    print(f"Task: {task['instruction']}")
    print(f"Domain: {task['web']}")
    print(f"Videos retrieved: {task['video_count']}")
    print(f"Planning knowledge: {task['planning_results'][:200]}...")
    print(f"Grounding knowledge: {task['grounding_results'][:200]}...")
```

Each entry contains:
- `id`: OSWorld task UUID
- `web`: Application domain
- `instruction`: Task instruction
- `query`: Generated search query for video retrieval
- `video_count` / `converted_video_count`: Number of videos retrieved/annotated
- `planning_results`: Full planning knowledge text (execution workflows, key considerations)
- `grounding_results`: Full grounding knowledge text (UI element catalog with visual descriptions)
- `cmd1_completed` / `cmd2_completed` / `cmd3_completed`: Pipeline stage completion flags

### Uploading to HuggingFace

```bash
# Prepare dataset directory
python scripts/prepare_hf_dataset.py --prepare \
    --videos_dir /path/to/videos \
    --urls_dir /path/to/urls \
    --converted_json /path/to/test_nogdrive_queries_with_videos_with_converted.json \
    --verification_json /path/to/video_verification_report.json \
    --output_dir ./hf_dataset

# Upload
python scripts/prepare_hf_dataset.py --upload \
    --dataset_dir ./hf_dataset \
    --repo_id sharryXR/GUIDE-dataset
```

---

## Project Structure

```
GUIDE/
├── guide/                                # Core GUIDE annotation pipeline
│   ├── youtube.py                        # Video-RAG: YouTube search, subtitle retrieval, 3-stage filtering
│   ├── asr.py                            # ASR: OpenAI Whisper (base), word-level timestamps
│   ├── run_sumvideo.py                   # Keyframe extraction: uniform sampling
│   ├── keyframe_subtitle.py              # Background subtraction: MOG2 with subtitle alignment
│   ├── action_annotation.py              # VLM action annotation: inverse dynamics on keyframe pairs
│   ├── action_annotation_prompt.py       # Annotation prompt templates (Meaningful filter, Thought & Action)
│   ├── auto_catch.py                     # Pipeline entry: ASR + keyframe extraction
│   ├── auto_work.py                      # Multi-stage orchestration: conda env switching, full pipeline
│   └── model/
│       └── load_model.py                 # Local VLM model loading (Qwen2.5-VL, Qwen3-VL)
│
├── osworld/                              # Complete OSWorld evaluation environment
│   ├── desktop_env/                      # VM environment: Docker providers, controllers, evaluators
│   ├── mm_agents/                        # Agent implementations
│   │   ├── agent.py                      # Base PromptAgent class
│   │   ├── prompts.py                    # Prompt templates for all agents
│   │   ├── seed_agent.py                 # Seed-1.8 agent + GUIDE Mode B integration
│   │   ├── qwen3vl_agent.py             # Qwen3-VL-8B agent + GUIDE Mode B integration
│   │   └── qwen25vl_agent.py            # Qwen2.5-VL agent
│   ├── new_gui_agents_with_video/        # AgentS3 framework + GUIDE Mode A integration
│   │   └── s3/
│   │       ├── agents/                   # agent_s.py, worker.py, grounding.py, code_agent.py
│   │       ├── core/                     # engine.py (LLM backends), mllm.py, module.py
│   │       ├── memory/                   # procedural_memory.py (all prompts)
│   │       ├── bbon/                     # behavior_narrator.py, comparative_judge.py
│   │       └── utils/                    # common_utils.py, formatters.py, local_env.py
│   ├── evaluation_examples/              # 361 task definitions across 10 domains
│   ├── run_local_withvideo.py            # AgentS3 + GUIDE experiment runner
│   ├── run_multienv_seed1_8.py           # Seed-1.8 + GUIDE experiment runner
│   ├── run_multienv_qwen3vl_full_annotation.py   # Qwen3-VL-8B + GUIDE runner
│   ├── run_multienv_qwen3vl_no_annotation.py     # Qwen3-VL-8B baseline (no GUIDE)
│   ├── video_self_convert.py             # Video annotation pipeline coordinator
│   ├── batch_convert_full_pipeline.py    # Batch annotation for all tasks
│   ├── show_result.py                    # Result aggregation and display
│   ├── lib_run_single.py                 # OSWorld single-task execution library
│   ├── lib_run_single_s3_with_video.py   # AgentS3 single-task execution with video
│   ├── run.py                            # Original OSWorld runner
│   ├── run_multienv.py                   # Original OSWorld multi-environment runner
│   ├── quickstart.py                     # OSWorld quick start example
│   ├── setup.py / pyproject.toml         # Package installation
│   ├── requirements.txt                  # OSWorld dependencies
│   └── configs/config.yaml               # Server configuration
│
├── scripts/
│   ├── prepare_hf_dataset.py             # HuggingFace dataset preparation and upload
│   └── HF_DATASET_CARD.md               # HuggingFace dataset card template
│
├── configs/config.yaml.example           # Configuration template
├── requirements.txt                      # GUIDE pipeline dependencies
├── .env.example                          # API key template
└── LICENSE                               # Apache License 2.0
```

---

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

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [OSWorld](https://github.com/xlang-ai/OSWorld) -- Evaluation benchmark (NeurIPS 2024)
- [OmniParser](https://github.com/microsoft/OmniParser) -- UI element detection
- [OpenAI Whisper](https://github.com/openai/whisper) -- Speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) -- Video downloading
