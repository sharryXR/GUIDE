# GUIDE Support for WindowsAgentArena

This directory contains the minimal public integration used to run GUIDE on
WindowsAgentArena (WAA). It does not include experiment outputs, downloaded
videos, logs, cookies, or private environment files.

## Contents

- `client/video_knowledge.py`: loads per-task GUIDE planning and grounding
  text for WAA tasks.
- `client/mm_agents/qwen3vl_vrag/`: Qwen3-VL WAA agent with GUIDE knowledge
  injection.
- `client/mm_agents/agent_s3/`: AgentS3 WAA wrapper with GUIDE knowledge
  injection.
- `tools/video_knowledge/`: utilities to build and backfill WAA video-knowledge
  indexes.
- `scripts/`: smoke-run helpers intended to run inside the WAA client container
  or development shell.

## Install Into A WAA Checkout

From the GUIDE repository root:

```bash
python windows_agent_arena/install_guide_waa.py \
  --waa-client-dir /path/to/WindowsAgentArena/src/win-arena-container/client \
  --install-agent-s3-backend \
  --patch-entrypoints \
  --overwrite
```

If you use a custom launcher instead of `start_client.sh`, the required Python
flags are `--video_json`, `--enable_planning`, `--enable_grounding`, and
`--grounding_max_k`.

## Build WAA Video Knowledge

Run from `WindowsAgentArena/src/win-arena-container/client`:

```bash
python /path/to/GUIDE/windows_agent_arena/tools/video_knowledge/build_waa_video_index.py \
  --waa-client-dir . \
  --osworld-converted /path/to/osworld_converted_video_knowledge.json \
  --output evaluation_examples_windows/test_all_queries_with_videos_with_converted.json
```

Optionally backfill WAA tasks that do not share an OSWorld canonical id by
matching similar annotated tasks:

```bash
python /path/to/GUIDE/windows_agent_arena/tools/video_knowledge/backfill_waa_video_index_by_similarity.py \
  --input-index evaluation_examples_windows/test_all_queries_with_videos_with_converted.json \
  --osworld-converted /path/to/osworld_converted_video_knowledge.json \
  --output evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json \
  --require-same-app-family
```

For missing tasks that need fresh web-video annotation, use the resumable
pipeline script and pass your GUIDE root explicitly:

```bash
python /path/to/GUIDE/windows_agent_arena/tools/video_knowledge/run_waa_gpt52_video_annotation.py \
  --index evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json \
  --output evaluation_examples_windows/test_all_queries_with_videos_with_similarity_gpt52.json \
  --video-root /path/to/GUIDE \
  --env-file /path/to/private/.env \
  --model gpt-5.2
```

The generated index JSON is a runtime artifact. Do not commit it unless it is a
sanitized release snapshot.

## Run

Inside the WAA client container or a WAA development shell where `/client`
points to `src/win-arena-container/client`:

```bash
export GUIDE_WAA_VIDEO_JSON=evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json
bash /path/to/GUIDE/windows_agent_arena/scripts/run_qwen3vl_vrag_smoke.sh
bash /path/to/GUIDE/windows_agent_arena/scripts/run_agents3_guide_smoke.sh
```

The scripts call `python run.py` directly. If you prefer WAA's shell launcher,
apply `start_client_guide_video.patch` and pass:

```bash
bash /start_client.sh \
  --agent qwen3vl_vrag \
  --model qwen3-vl-plus \
  --video-json evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json \
  --enable-planning true \
  --enable-grounding true \
  --grounding-max-k 8
```

## Notes

- Raw videos/subtitles/cookies are intentionally excluded.
- Release snapshots should contain only permitted metadata such as selected
  video IDs, URLs, titles, cached annotations, and benchmark task snapshots.
- The AgentS3 wrapper expects GUIDE's `new_gui_agents_with_video` package to be
  installed into the WAA client; the installer can copy it with
  `--install-agent-s3-backend`.
