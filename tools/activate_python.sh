#!/usr/bin/env bash
# THIS FILE IS GENERATED BY tools/setup_anaconda.sh
if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /N/slate/ak16/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
