#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: aggressive run aiming to break past val AUC 0.860 ----
#
# Stack (vs the previous 0.809 leaderboard run):
#   * seq_top_k 64 -> 128: doubles the recent-behavior context kept by
#     LongerEncoder. Compression remains as implicit regularization.
#   * --use_rope: relative position encoding for sequence attention.
#   * --use_swa --swa_start_epoch 2: average epoch>=2 weights into a flat
#     minimum for the final checkpoint; targets the val->test gap.
#   * dropout 0.15 -> 0.18, weight_decay 1e-5 -> 3e-5: more regularization
#     to keep that gap from widening as we capture more signal.
#   * num_epochs 4 -> 6, patience 2 -> 3: give SWA at least 4-5 epochs of
#     averaging to actually do something.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 4 \
    --seq_encoder_type longer \
    --seq_top_k 128 \
    --reinit_cardinality_threshold 0 \
    --patience 3 \
    --dropout_rate 0.18 \
    --num_epochs 6 \
    --weight_decay 3e-5 \
    --lr_schedule cosine \
    --use_rope \
    --use_swa \
    --swa_start_epoch 2 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
