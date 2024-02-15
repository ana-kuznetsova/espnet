#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_sets="train_clean_100 train_clean_360 train_other_500"
dev_sets="dev_clean dev_other"
test_sets="test_clean test_other"
DATA_ROOT="/data/anakuzne/espnet/egs2/librispeech/asr1/data"
DUMP_DIR="/data/anakuzne/espnet/egs2/librispeech/asr1/dump_codec/extracted/"

echo "Preparing training set..."

#for dset in ${train_sets}; do
#    echo "Dataset ${dset}"
#    scripts/audio/format_wav_scp.sh ${DATA_ROOT}/${dset}/wav.scp ${DUMP_DIR}/${dset}
#    python pyscripts/feats/dump_ssl_feature.py \
#             --feature_conf "/data/anakuzne/espnet/egs2/librispeech_100/asr1/conf/tuning/enc_asr/feats.yaml" \
#             --in_filetype "sound" \
#             --utt2num_samples "${DUMP_DIR}/${dset}/utt2num_samples" \
#             --out_filetype "mat" \
#             "scp:${DUMP_DIR}/${dset}/wav.scp" \
#            "ark,scp:${DUMP_DIR}/${dset}/feats.ark,${DUMP_DIR}/${dset}/feats.scp"
#done
for dset in ${train_sets}; do
    cp ${DATA_ROOT}/${dset}/utt2spk ${DUMP_DIR}/${dset}
done

echo "Combining datasets -> train_960"
utils/data/combine_data.sh ${DUMP_DIR}/train_960 ${DUMP_DIR}/train_clean_100 ${DUMP_DIR}/train_clean_360 ${DUMP_DIR}/train_other_500
utils/data/fix_data_dir.sh ${DUMP_DIR}/train_960

for dset in ${train_sets}; do
    utils/data/fix_data_dir.sh ${DUMP_DIR}/${dset}
done

echo "Preparing dev set... "

for dset in ${dev_sets}; do
    echo "Dataset ${dset}"
    scripts/audio/format_wav_scp.sh ${DATA_ROOT}/${dset}/wav.scp ${DUMP_DIR}/${dset}
    python pyscripts/feats/dump_ssl_feature.py \
             --feature_conf "/data/anakuzne/espnet/egs2/librispeech_100/asr1/conf/tuning/enc_asr/feats.yaml" \
             --in_filetype "sound" \
             --utt2num_samples "${DUMP_DIR}/${dset}/utt2num_samples" \
             --out_filetype "mat" \
             "scp:${DUMP_DIR}/${dset}/wav.scp" \
            "ark,scp:${DUMP_DIR}/${dset}/feats.ark,${DUMP_DIR}/${dset}/feats.scp"
done

echo "Combining datasets -> dev"
utils/data/combine_data.sh ${DUMP_DIR}/dev ${DUMP_DIR}/dev_clean ${DUMP_DIR}/dev_other
utils/data/fix_data_dir.sh ${DUMP_DIR}/dev

for dset in ${dev_sets}; do
    utils/data/fix_data_dir.sh ${DUMP_DIR}/${dset}
done

echo "Preparing test sets ... "

for dset in ${test_sets}; do
    echo "Dataset ${dset}"
    scripts/audio/format_wav_scp.sh ${DATA_ROOT}/${dset}/wav.scp ${DUMP_DIR}/${dset}
    python pyscripts/feats/dump_ssl_feature.py \
             --feature_conf "/data/anakuzne/espnet/egs2/librispeech_100/asr1/conf/tuning/enc_asr/feats.yaml" \
             --in_filetype "sound" \
             --utt2num_samples "${DUMP_DIR}/${dset}/utt2num_samples" \
             --out_filetype "mat" \
             "scp:${DUMP_DIR}/${dset}/wav.scp" \
            "ark,scp:${DUMP_DIR}/${dset}/feats.ark,${DUMP_DIR}/${dset}/feats.scp"
done

for dset in ${test_sets}; do
    utils/data/fix_data_dir.sh ${DUMP_DIR}/${dset}
done