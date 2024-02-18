#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"

asr_config=conf/tuning/enc_asr/train_asr_linear_encoder.yaml
inference_config=conf/decode_asr.yaml
#dump_dir=/data/anakuzne/espnet/egs2/librispeech_100/asr1/dump_codec_3kbps
#export PATH=$PATH:/data/anakuzne/espnet/kaldi/tools/sctk/src/sclite

./asr.sh \
    --lang en \
    --asr_tag codec_frozen_linear_sp_with_lm_$(date -I) \
    --stage 10 \
    --ngpu 1 \
    --nj 11 \
    --gpu_inference true \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --inference_nj 11\
    --nbpe 5000 \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
