"""
Author: Anurag Kumar
"""
import os
from shutil import copyfile
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool, Manager, Process, RLock


def calc_CR_MLS(pid, data_dir, map_, file_, start=None, end=None):
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    files = map_
    data = file_[start:end]
    p = os.path.join(data_dir,'audio')
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for _, row in enumerate(data):
            fpath = '/'.join(row.split('_')[1:-1]) + row.split(' ')[0][4:] + '.flac'
            print(fpath)
            fname_in = os.path.join(p, fpath)
            fname_out = os.path.join('/shared/workspaces/anuragkumar95/compressions/',filename)
            temp = subprocess.run(["ffmpeg","-i", 
                                   fname_in, fname_out[:-5]+".wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            temp = subprocess.run(["gzip", "-k", fname_out[:-5]+".wav"])
            fsize = os.path.getsize(fname_out[:-5]+".wav")
            fsize_comp = os.path.getsize(fname_out[:-5]+".wav.gz")
            temp = subprocess.run(["rm", fname_out[:-5]+".wav.gz"])
            temp = subprocess.run(["rm", fname_out[:-5]+".wav"])
            try:
                CR = fsize_comp/fsize
            except Exception as e:
                print(f"File: {fname}, Ori:{fsize}, Compr:{fsize_comp}")
                print(e)
                raise ZeroDivisionError
            files['mls_'+filename.split('.')[0]] = str(CR)
            pbar.update(1)

def calc_CR_CV(pid, data_dir, map_, file_, start=None, end=None):
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    p = os.path.join(data_dir,'clips')
    files = map_
    data = file_[start:end]
    #for idx, row in tqdm(data.iterrows()): 
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for idx, row in data.iterrows():
            fname = row['path']
            fname_in = os.path.join(p, fname)
            temp = subprocess.run(["sox", 
                                   "--bits", "32", 
                                   "--channels", "1", 
                                   "--encoding","signed-integer",
                                   "--rate","48000",
                                   fname_in, fname_in[:-4]+".wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            temp = subprocess.run(["gzip", "-k", fname_in[:-4]+".wav"])
            fsize = os.path.getsize(fname_in[:-4]+".wav")
            fsize_comp = os.path.getsize(fname_in[:-4]+".wav"+'.gz')
            temp = subprocess.run(["rm", fname_in[:-4]+".wav"+".gz"])
            temp = subprocess.run(["rm", fname_in[:-4]+".wav"])
            try:
                CR = fsize_comp/fsize
            except Exception as e:
                print(f"File: {fname}, Ori:{fsize}, Compr:{fsize_comp}")
                print(e)
                raise ZeroDivisionError
            files[fname.split('.')[0]] = str(CR)
            pbar.update(1)


def save_file(map_, res_dir, db, wav_scp=None, compression=None): 
    print('\n') 
    if not compression:
        with open(os.path.join(res_dir, "compression_ratio_"+db), 'w') as fo:
            if wav_scp:
                fe = open(res_dir+'/extras', 'w') 
                print("Comparing wav_scp files.....")
                for line in tqdm(open(wav_scp,'r').readlines()):
                    fname = line.split()[0]
                    #if fname in map_:
                    fo.write(fname + ' ' + map_[fname]+'\n')
                    #else:
                    #    fe.write(line+'\n')
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
    if args.db == 'cv':
        files = ['validated.tsv', 'invalidated.tsv']
    if args.db == 'mls':
        files = ['mls_files.tsv']
    for file_ in files: 
        pool = Pool(processes=args.num_process, initargs=(RLock(), ), initializer=tqdm.set_lock)
        processes = []
        csv_path = os.path.join(args.data_dir, file_)
        csv = pd.read_csv(csv_path, sep = '\t')
        if args.db == 'mls':
            csv = open(args.wav_scp, 'r').readlines()
        csv_len = 100#len(csv)
        rows_per_process = int(csv_len/args.num_process) + 1
        print('\n')
        print(f"starting processes for file {file_} with {rows_per_process} rows")
        for i in range(args.num_process):
            start = int(i*rows_per_process)
            end = int(min(start + rows_per_process, csv_len))
            if args.db == 'cv':
                processes.append(pool.apply_async(calc_CR_CV, args=(i,
                                    args.data_dir, 
                                    map_, 
                                    csv, 
                                    start,
                                    end,)))
            if args.db == 'mls':
                processes.append(pool.apply_async(calc_CR_MLS, args=(i,
                                    args.data_dir, 
                                    map_, 
                                    csv, 
                                    start,
                                    end,)))

        pool.close()
        results = [job.get() for job in processes]
    if args.wav_scp:
        save_file(map_, args.res_dir, args.db, args.wav_scp)
    else:
        save_file(map_, args.res_dir,  args.db)
    print("compression ratio file created successfully...")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help='Type of dataset, cv(commonvoice) or mls')
    parser.add_argument("--num_process", type=int, required=False)
    parser.add_argument('--wav_scp', type=str, required=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')
    args = parser.parse_args()
    main(args)


    