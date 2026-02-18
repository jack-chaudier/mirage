#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
  echo "Could not find .venv Python. Expected ${REPO_ROOT}/.venv/bin/python"
  exit 1
fi

MODEL_ID="mlx-community/gemma-2-2b-it-4bit"
DATA_DIR="${SCRIPT_DIR}/training_data"
ADAPTER_DIR="${SCRIPT_DIR}/adapters/mirage_aware_v2"
RESULTS_CSV="${SCRIPT_DIR}/results/mirage_aware_eval_results.csv"
SUMMARY_CSV="${SCRIPT_DIR}/results/mirage_aware_eval_summary.csv"

run_train() {
  local num_layers="$1"
  local iters="$2"

  "${PYTHON_BIN}" -m mlx_lm lora \
    --model "${MODEL_ID}" \
    --data "${DATA_DIR}" \
    --train \
    --batch-size 1 \
    --max-seq-length 2560 \
    --num-layers "${num_layers}" \
    --iters "${iters}" \
    --learning-rate 1e-4 \
    --steps-per-eval 50 \
    --adapter-path "${ADAPTER_DIR}" \
    --seed 42
}

echo "Using Python: ${PYTHON_BIN}"
echo "=== Step 0: Install/verify dependencies ==="
"${PYTHON_BIN}" -m pip install -U mlx-lm numpy pandas requests scipy scikit-learn tqdm
"${PYTHON_BIN}" -m mlx_lm lora --help >/dev/null

echo "=== Step 0b: Verify MLX JSONL parser accepts extra metadata fields ==="
"${PYTHON_BIN}" - <<'PY'
import types
from mlx_lm.tuner.datasets import create_dataset

class DummyTokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, return_dict=False):
        text = " ".join(m.get("content", "") for m in messages)
        return [1] * max(1, len(text.split()))

row = {
    "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
    "oracle_pivot": "B01-E001",
    "evidence_degraded": False,
    "prereq_ratio": 1.0,
    "difficulty": "easy",
}
cfg = types.SimpleNamespace(
    mask_prompt=False,
    chat_feature="messages",
    prompt_feature="prompt",
    completion_feature="completion",
    text_feature="text",
)
ds = create_dataset([row], DummyTokenizer(), cfg)
ds.process(ds[0])
print("MLX dataset preflight passed.")
PY

echo "=== Step 1: Generate training data ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_training_data.py" \
  --output-dir "${DATA_DIR}" \
  --categories investment,incident,narrative \
  --compression-levels 0.4,0.5,0.6 \
  --seeds 101,202,303 \
  --target-prereq-ratios 0.0,0.25,0.5,0.75 \
  --max-context-words 1200 \
  --split-seed 42 \
  --no-balance-labels \
  --balance-seed 42

echo "=== Step 2: Fine-tune (primary: num_layers=8, iters=600) ==="
if run_train 8 600; then
  echo "Primary training completed."
else
  echo "Primary training failed. Retrying with num_layers=4, iters=600..."
  if run_train 4 600; then
    echo "Fallback training completed."
  else
    echo "Fallback failed. Final retry with num_layers=4, iters=400..."
    run_train 4 400
  fi
fi

echo "=== Step 3: Evaluate ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/eval_mirage_aware.py" \
  --model "${MODEL_ID}" \
  --adapter-path "${ADAPTER_DIR}" \
  --valid-jsonl "${DATA_DIR}/valid.jsonl" \
  --out-csv "${RESULTS_CSV}" \
  --summary-csv "${SUMMARY_CSV}" \
  --max-new-tokens 280 \
  --temperature 0.0 \
  --print-samples 5

echo "=== Done ==="
echo "Adapter: ${ADAPTER_DIR}"
echo "Eval results: ${RESULTS_CSV}"
echo "Eval summary: ${SUMMARY_CSV}"
