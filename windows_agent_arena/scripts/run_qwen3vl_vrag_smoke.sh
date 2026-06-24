#!/usr/bin/env bash
set -euo pipefail

WAA_CLIENT_DIR=${WAA_CLIENT_DIR:-/client}
VIDEO_JSON=${GUIDE_WAA_VIDEO_JSON:-evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json}
JSON_NAME=${GUIDE_WAA_JSON_NAME:-evaluation_examples_windows/test_all.json}
MODEL=${GUIDE_WAA_QWEN_MODEL:-qwen3-vl-plus}

cd "$WAA_CLIENT_DIR"

python run.py \
  --agent_name qwen3vl_vrag \
  --model "$MODEL" \
  --test_all_meta_path "$JSON_NAME" \
  --video_json "$VIDEO_JSON" \
  --enable_planning \
  --enable_grounding \
  --grounding_max_k "${GUIDE_WAA_GROUNDING_MAX_K:-8}" \
  --max_tokens "${GUIDE_WAA_MAX_TOKENS:-8192}" \
  --max_steps "${GUIDE_WAA_MAX_STEPS:-50}" \
  --result_dir "${GUIDE_WAA_RESULT_DIR:-./results_qwen3vl_vrag}"
