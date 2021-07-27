import os
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
"""
def calc_CR(wav_scp, data_dir, res_dir):

    with open(wav_scp) as fo:
        commands = fo.readlines()

    with open(os.path.join(res_dir, "compression_ratio"), 'w') as fo:
        for cmd in tqdm(commands):
            print("Command:", cmd)
            fname = cmd.split()[0]
            fpath = os.path.join(data_dir, "/".join(fname.split('_')[:-1]), fname)
            print("fpath:", fpath)
            cmd_convert = cmd.split()[1:-2]
            print(cmd_convert)
            cmd_convert.pop(5)
            cmd_convert.insert(5, fpath+'.opus')
            cmd_convert.append(os.path.join(res_dir, fname+'.wav'))
            cmd_convert = subprocess.run(cmd_convert, stdout=subprocess.PIPE, 
                                                    text=True, check=True)
            fname_out = os.path.join(res_dir, fname)
            temp = subprocess.run(["gzip", "-k", fname_out+'.wav'])
            fsize = subprocess.run(["du", fname_out+'.wav'], stdout=subprocess.PIPE, 
                                                text=True, check=True)
            fsize_comp = subprocess.run(["du", fname_out+'.wav'+'.gz'], stdout=subprocess.PIPE, 
                                                text=True, check=True)
            fsize = int(fsize.stdout.split('\t')[0])
            fsize_comp = int(fsize_comp.stdout.split('\t')[0])
            temp = subprocess.run(["rm", fname_out+".wav"+".gz"])
            temp = subprocess.run(["rm", fname_out+".wav"])
            CR = 1 - (fsize_comp/fsize)

            fo.write(fname+' '+str(CR)+"\n")
"""
def calc_CR(data_dir, res_dir):
    p = os.path.join(data_dir,'clips')
    train_csv = os.path.join(data_dir, 'train.tsv')
    #data = pd.read_csv(train_csv, sep='\t')['path']
    data = pd.read_csv(train_csv, sep='\t')
    with open(os.path.join(res_dir, "compression_ratio"), 'w') as fo:
        for ind, row in tqdm(data.iterrows()):
            fname = row['path']
            client = row['client_id']
            fname_in = os.path.join(p, fname)
            print(fname_in)
            temp = subprocess.run(["gzip", "-k", fname_in])
            fsize = subprocess.run(["du", fname_in], stdout=subprocess.PIPE, 
                                                text=True, check=True)
            fsize_comp = subprocess.run(["du", fname_in+'.gz'], stdout=subprocess.PIPE, 
                                                text=True, check=True)
            fsize = int(fsize.stdout.split('\t')[0])
            fsize_comp = int(fsize_comp.stdout.split('\t')[0])
            temp = subprocess.run(["rm", fname_in+".gz"])
            CR = 1 - (fsize_comp/fsize)
            fo.write(client+'-'+fname.split('.')[0]+' '+"["+str(CR)+"]\n")
            fo.write('sp0.9-'+client+'-'+fname.split('.')[0]+' '+"["+str(CR)+"]\n")
            fo.write('sp1.1-'+client+'-'+fname.split('.')[0]+' '+"["+str(CR)+"]\n")
            



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_scp', type=str, required=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')

    args = parser.parse_args()
    calc_CR(args.data_dir, args.res_dir)