#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"
#test_sets="dev_clean dev_other"

asr_config=conf/tuning/enc_asr/train_asr_linear_encodec_encoder.yaml
inference_config=conf/decode_asr.yaml
dump_dir=/data/anakuzne/espnet/egs2/librispeech_100/asr1/dump_encodec
stats_dir=asr_stats_raw_en_bpe5000_sp_encodec

./asr.sh \
    --lang en \
    --asr_tag codec_frozen_linear_encodec_lr_2e-5_sp_quantizer_true_2024-02-28 \
    --stage 11 \
    --ngpu 1 \
    --nj 1 \
    --gpu_inference true \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --inference_nj 1 \
    --asr_stats_dir "${stats_dir}" \
    --nbpe 5000 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --dumpdir "${dump_dir}" \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"