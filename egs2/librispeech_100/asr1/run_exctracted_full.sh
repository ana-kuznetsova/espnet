#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"
#test_sets="dev_clean dev_other"

asr_config=conf/tuning/enc_asr/train_asr_conformer_nondet_codec_full.yaml
inference_config=conf/decode_asr.yaml
dump_dir=/data/anakuzne/espnet/egs2/librispeech_100/asr1/dump_codec
export PATH=$PATH:/data/anakuzne/espnet/kaldi/tools/sctk/src/sclite
./asr.sh \
    --lang en \
    --asr_tag asr_codec_frozen_from_pretrained_decoder_6layer_frozen_enc_reduced_conformer_lr2e-2_$(date -I) \
    --stage 11 \
    --ngpu 1 \
    --nj 11 \
    --gpu_inference true \
    --inference_nj 11 \
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
