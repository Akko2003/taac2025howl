#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: DCN-V2 added to break past val AUC 0.860 ceiling ----
#
# Carries forward the previous run's stack (SWA + RoPE + cosine LR + tight
# regularization). New here:
#   * --use_dcn_v2 --dcn_v2_layers 3: parallel cross network on the
#     flattened gated NS tokens + raw dense features. Adds explicit
#     polynomial-style feature crosses that the deep/attention towers
#     would otherwise have to discover from scratch. The cross logit head
#     is zero-initialized so the model boots equivalent to the previous
#     deep-only network and only diverges as the cross branch learns
#     useful interactions.
#   * --dcn_v2_low_rank 0: full-rank square W_l per layer. Switch to
#     a positive value (e.g. 64) if memory becomes tight.
#
# SWA window slightly extended (start_epoch 2->3, num_epochs 6->5) so the
# DCN-V2 branch has 1-2 extra epochs of solo training before being folded
# into the running mean.
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
    --num_epochs 5 \
    --weight_decay 3e-5 \
    --lr_schedule cosine \
    --use_rope \
    --use_swa \
    --swa_start_epoch 3 \
    --use_dcn_v2 \
    --dcn_v2_layers 3 \
    --dcn_v2_low_rank 0 \
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
