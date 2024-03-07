#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"

asr_config=conf/tuning/enc_asr/train_asr_encodec_frozen_linear_preenc_full_encoder.yaml
inference_config=conf/decode_asr.yaml
#dump_dir=/data/anakuzne/espnet/egs2/librispeech_100/asr1/dump_codec_3kbps
#export PATH=$PATH:/data/anakuzne/espnet/kaldi/tools/sctk/src/sclite

#    --speed_perturb_factors "0.9 1.0 1.1" \

./asr.sh \
    --lang en \
    --asr_tag codec_frozen_dac_full_conformer_no_init_no_layer_norm_warmup_lr_sp_$(date -I) \
    --stage 11 \
    --ngpu 2 \
    --nj 1 \
    --gpu_inference true \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --inference_nj 1 \
    --nbpe 5000 \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
