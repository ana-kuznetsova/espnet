"""
Author: Anurag Kumar
"""
import os
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool, Manager, Process, RLock

def calc_CR(pid, data_dir, res_dir, map_, file_, start=None, end=None):
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    p = os.path.join(data_dir,'clips')
    files = map_
    data = file_[start:end]
    #for idx, row in tqdm(data.iterrows()): 
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for idx, row in data.iterrows():
            fname = row['path']
            client = row['client_id']
            utt = row['sentence']
            fname_in = os.path.join(p, fname)
            temp = subprocess.run(["sox", 
                                   "--bits", "32", 
                                   "--channels", "1", 
                                   "--encoding","signed-integer",
                                   "--rate","48000",
                                   fname_in, "".join(fname_in.split('.')[:-1])+".wav"], stdout=subpro$
            temp = subprocess.run(["gzip", "-k", "".join(fname_in.split('.')[:-1])+".wav"])
            #fsize = subprocess.run(["du", fname_in.split('.')[0]+".wav"], stdout=subprocess.PIPE, 
            #                                    text=True, check=True)
            #fsize_comp = subprocess.run(["du", fname_in.split('.')[0]+".wav"+'.gz'], stdout=subproce$
            #                                    text=True, check=True)
            fsize = os.path.getsize("".join(fname_in.split('.')[:-1])+".wav")
            fsize_comp = os.path.getsize("".join(fname_in.split('.')[:-1])+".wav"+'.gz')
            #fsize = int(fsize.stdout.split('\t')[0])
            #fsize_comp = int(fsize_comp.stdout.split('\t')[0])
            temp = subprocess.run(["rm", "".join(fname_in.split('.')[:-1])+".wav"+".gz"])
            temp = subprocess.run(["rm", "".join(fname_in.split('.')[:-1])+".wav"])
            CR = fsize_comp/fsize
            files[client+'-'+fname.split('.')[0]] = str(CR)
            pbar.update(1)


def save_file(map_, res_dir, wav_scp=None, compression=None): 
    print('\n') 
    if not compression:
        with open(os.path.join(res_dir, "compression_ratio"), 'w') as fo:
            if wav_scp:
                fe = open(res_dir+'/extras', 'w') 
                print("Comparing wav_scp files.....")
                for line in tqdm(open(wav_scp,'r').readlines()):
                    fname = line.split()[0]
                    if fname in map_:
                        fo.write(fname + ' ' + map_[fname]+'\n')
                    else:
                        fe.write(line+'\n')
                fe.close()
            else:
                for file in map_:
                    fo.write(file + ' ' + map_[file]+'\n')
    else:
        compressions = {}
        wavs = {i.split()[0]: 1 for i in open(wav_scp,'r').readlines()}
        #print(list(wavs.keys())[:10])
        for line in tqdm(open(compression, 'r').readlines()):
            #print(line)
            fname = line.split()[0]
            cr = line.split()[1]
            #print("FNAME:",fname, "CR:",cr)
            if fname in wavs:
                compressions[fname] = cr
        
        with open(compression+'_new', 'w') as fo:
            for fname in compressions:
                fo.write(fname+' '+compressions[fname]+'\n')

def main(args):
    manager = Manager()
    map_ = manager.dict()
    if not args.compression:
        if not args.num_process:
            calc_CR(args.data_dir, args.res_dir, map_) 
        else:
            for file_ in ['validated.tsv', 'invalidated.tsv']: 
                pool = Pool(processes=args.num_process, initargs=(RLock(), ), initializer=tqdm.set_lock)
                processes = []
                csv_path = os.path.join(args.data_dir, file_)
                csv = pd.read_csv(csv_path, sep = '\t')
                csv_len = len(csv)
                rows_per_process = int(csv_len/args.num_process) + 1
                print('\n')
                print(f"starting processes for file {file_} with {rows_per_process} rows")
                for i in range(args.num_process):
                    start = int(i*rows_per_process)
                    end = int(min(start + rows_per_process, csv_len))
                    processes.append(pool.apply_async(calc_CR, args=(i,
                                        args.data_dir, 
                                        args.res_dir, 
                                        map_, 
                                        csv, 
                                        start,
                                        end,)))
        
                pool.close()
                results = [job.get() for job in processes]
    if args.wav_scp:
        save_file(map_, args.res_dir, args.wav_scp, args.compression)
    else:
        save_file(map_, args.res_dir)
    print("compression ratio file created successfully...")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, required=False)
    parser.add_argument('--wav_scp', type=str, required=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')
    parser.add_argument("--compression", type=str, required=False, help='Path to compression file')
    args = parser.parse_args()
    main(args)