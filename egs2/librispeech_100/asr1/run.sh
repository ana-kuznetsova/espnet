#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/enc_asr/train_asr_dac_frozen_decode_melspec_codebooks.yaml
inference_config=conf/decode_asr.yaml
dump_dir="dump_codec_full"
#stats_dir=asr_stats_raw_en_bpe5000_sp_encodec

./asr.sh \
    --lang en \
    --asr_tag codec_frozen_decode_true_codebooks_1_conv2d_embeds_no_init_$(date -I) \
    --stage 11 \
    --ngpu 1 \
    --nj 1 \
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
    --bpe_train_text "data/${train_set}/text" "$@"
