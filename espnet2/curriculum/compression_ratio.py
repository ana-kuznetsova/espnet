import os
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
import threading
"""
def calc_CR_wav(wav_scp, data_dir, res_dir):

    with open(wav_scp) as fo:
        commands = fo.readlines()

    with open(os.path.join(res_dir, "compression_ratio"), 'w') as fo:
        for cmd in tqdm(commands):
            print("Command:", cmd)
            fname = cmd.split()[0]
            fpath = os.path.join(data_dir, "/".join(fname.split('-')[:-1]), fname)
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
def calc_CR(data_dir, res_dir, map_, file_, start=None, end=None):
    p = os.path.join(data_dir,'clips')
    files = map_
    data = file_[start:end]
    for x in tqdm(data.iterrows()):
        ind, row = x
        fname = row['path']
        client = row['client_id']
        fname_in = os.path.join(p, fname)
        temp = subprocess.run(["gzip", "-k", fname_in])
        fsize = subprocess.run(["du", fname_in], stdout=subprocess.PIPE, 
                                            text=True, check=True)
        fsize_comp = subprocess.run(["du", fname_in+'.gz'], stdout=subprocess.PIPE, 
                                            text=True, check=True)
        fsize = int(fsize.stdout.split('\t')[0])
        fsize_comp = int(fsize_comp.stdout.split('\t')[0])
        temp = subprocess.run(["rm", fname_in+".gz"])
        CR = 1 - (fsize_comp/fsize)
        files[client+'-'+fname.split('.')[0]] = str(CR)


def save_file(map_, res_dir, wav_scp=None):  
    with open(os.path.join(res_dir, "compression_ratio"), 'w') as fo:
        if wav_scp:
            fe = open('extras', 'w') 
            print("Comapring wav_scp files.....")
            for line in open(wav_scp,'r').readlines():
                fname = line.split()[0]
                if fname in files:
                    fo.write(fname + ' ' + map_[fname]+'\n')
                else:
                    fe.write(fname)
            fe.close()
        else:
            for file in map_:
                fo.write(file + ' ' + map_[file]+'\n')

                #fo.write('sp0.9-'+client+'-'+fname.split('.')[0]+' '+"["+str(CR)+"]\n")
                #fo.write('sp1.1-'+client+'-'+fname.split('.')[0]+' '+"["+str(CR)+"]\n")
            
def main(args):
    map_ = {}
    if not args.num_threads:
        calc_CR(args.data_dir, args.res_dir, map_) 
    else:
        for file_ in ['validated.tsv', 'invalidated.tsv']: 
            threads = []
            csv_path = os.path.join(args.data_dir, file_)
            csv = pd.read_csv(csv_path, sep = '\t')
            csv_len = len(csv)
            rows_per_thread = csv_len/args.num_threads
            for i in range(args.num_threads):
                t = threading.Thread(target=calc_CR, args=(args.data_dir, 
                                                           args.res_dir, 
                                                           map_, 
                                                           csv, 
                                                           int(i*rows_per_thread),
                                                           int(min((i+1)*rows_per_thread, csv_len)), ))
                threads.append(t)
                print(f"starting thread {i} for file {file_}")
                t.start()
            print(f"waiting for threads to finish for file {file_}")
            for thread in threads:
                thread.join()

    if args.wav_scp:
        save_file(map_, args.res_dir, args.wav_scp)
    else:
        save_file(map_, args.res_dir)
    print("compression ratio file created successfully...")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, required=False)
    parser.add_argument('--wav_scp', type=str, required=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')

    args = parser.parse_args()
    main(args)