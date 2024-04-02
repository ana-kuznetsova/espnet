#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="debug_subset_train"
valid_set="debug_subset_dev"
test_sets="debug_subset_test"

asr_config=conf/tuning/enc_asr/debug_scheduler_conf.yaml
inference_config=conf/decode_asr.yaml
dump_dir="dump_debug_subset"
stats_dir="asr_debug_subset_bpe5000_sp"
#Codec_base tag has dim 256 for attention

./asr.sh \
    --lang en \
    --asr_tag debug_new_scheduler_$(date -I) \
    --asr_stats_dir "${stats_dir}" \
    --stage 11 \
    --ngpu 1 \
    --nj 1 \
    --local_data_opts true \
    --dumpdir "${dump_dir}" \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --lm_tag "lm_debug_subset"\
    --lm_exp "lm_stats_debug_subset_en_bpe5000"\
    --bpe_train_text "data/${train_set}/text" "$@"
