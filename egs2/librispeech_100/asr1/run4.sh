#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=/data/anakuzne/espnet/egs2/librispeech_100/asr1/conf/tuning/enc_asr/svd_vq_conformer.yaml
inference_config=conf/decode_asr.yaml
dump_dir="dump_base"
stats_dir="asr_base_stats_bpe5000_sp"
#Codec_base tag has dim 256 for attention

./asr.sh \
    --lang en \
    --local_data_opts false \
    --asr_tag conformer_svd_dev_$(date -I) \
    --asr_stats_dir "${stats_dir}" \
    --stage 10 \
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
