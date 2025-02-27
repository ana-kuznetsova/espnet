key_file="/shared/50k_train/mls_english_opus/dump/extracted/train_norm_debug/wav.scp"
_nj=10
split_scps=""
_logdir="/shared/workspaces/anakuzne/tmp/log"
_inp_dir="/shared/workspaces/anakuzne/tmp/inp_scp"
_out_dir="/shared/workspaces/anakuzne/tmp/test_files"
format='scp'

for n in $(seq ${_nj}); do
    split_scps+=" ${_inp_dir}/train.${n}.scp"
done

echo "Split files..."                                                                                                                                                                                                           
/shared/50k_train/mls_english_opus/utils/split_scp.pl "${key_file}" ${split_scps}




echo "Calculate complexity..."
/shared/50k_train/mls_english_opus/utils/queue.pl JOB=1:"${_nj}" "${_logdir}"/train.JOB.scp

if [ $format == "scp" ]; then
    ./shared/workspaces/jp/rev-kaldi/src/featbin/compute-comp-ratio scp,p:${_inp_dir}/train.JOB.scp \
                                                                ark,t:${_out_dir}/comp_ratio.JOB.txt
else
    ./shared/grid_setup/rev-kaldi/src/featbin/extract-segments scp,p:${_inp_dir}/train.JOB.scp test.segments ark:-  |  ./compute-comp-ratio ark:- ark,t:test.output.txt