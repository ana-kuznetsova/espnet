#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/enc_asr/train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic.yaml
inference_config=conf/decode_asr.yaml

#Add path to sclite for scoring
export PATH=$PATH:/data/anakuzne/espnet/kaldi/tools/sctk/src/sclite

./asr.sh \
    --lang en \
    --asr_tag base_no_spec_augment_1024_enc_out \
    --stage 1 \
    --ngpu 1 \
    --nj 1 \
    --gpu_inference true \
    --inference_nj 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
