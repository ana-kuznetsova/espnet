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
dump_dir=/data/anakuzne/espnet/egs2/librispeech_100/asr1/dump_codec_3kbps
#export PATH=$PATH:/data/anakuzne/espnet/kaldi/tools/sctk/src/sclite

./asr.sh \
    --lang en \
    --asr_tag codec_frozen_from_pretrained_3kbps_lr1e-4_$(date -I) \
    --stage 2 \
    --ngpu 1 \
    --nj 11 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe 5000 \
    --audio_format "flac.ark" \
    --feats_type extracted \
    --dumpdir "${dump_dir}" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
