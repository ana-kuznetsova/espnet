"""
Author: Anurag Kumar
"""

import os
import glob
from shutil import copyfile
import pandas as pd
from pydub import AudioSegment
from pathlib import Path
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool, Manager, Process, RLock

def convert_to_wav(fin, fout):
    """
    Reads the file from fin and saves the file in wav format in fout
    """
    temp = subprocess.run(["ffmpeg",
                           "-i", 
                           fin, 
                           fout], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
    
    
def compress_segments(map_, wav_id, file_path, segments, outpath):
    """
    Accepts the wav audio file and list of segment time stamps.
    Chunks the audio and calculates CR for each segment.
    
    file_path : path to wav file
    segments  : list  of segments time stamps
    outpath   : path to save chunks
    """
    try:
        audio = AudioSegment.from_wav(file_path)
        #print("\nSegments:", len(segments))
        for _, row in segments.iterrows():
            start = row[2] * 1000
            end = row[3] * 1000
            audio_chunk = audio[start:end]
            save_path = "{}/{}_chunk_{}_{}.wav".format(outpath, wav_id, start, end)
            audio_chunk.export(save_path, format='wav')
            compress_file(map_=map_, 
                        name=row[0],
                        save_path=save_path)
    except Exception as e:
        print("ERR:",e)
        print("Failed files:", file_path)

def compress_file(map_, name, save_path):
    """
    Compresses the file and calculates CR.

    map_: common map {uttid : CR} across processes
    name: utt_id
    save_path: temp path to store converted wavs/ compressed files.
    """
    size = os.path.getsize(save_path)
    temp = subprocess.run(["gzip", "-k", save_path])
    cr_size = os.path.getsize(save_path+".gz")
    try:
        map_[name] = cr_size / size
    except Exception as e:
        print(f"File: {save_path}, Ori:{size}, Compr:{cr_size}")
        print(e)
        raise ZeroDivisionError
    temp = subprocess.run(["rm", save_path])
    temp = subprocess.run(["rm", save_path+".gz"])


def calc_CR_scp(pid, map_, file_, args, segments=None, start=None, end=None):
    """
    Main function run by each process.

    pid     : process_id
    map_    : common map {uttid : CR} across processes
    file_   : entire wav.scp to be read
    segments: segment file if exists
    start   : start index for process
    end     : end index for process
    """
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    data = file_[start:end]
    if segments:
        segments = pd.read_csv(segments, sep = ' ', header=None)
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for row in data:
            sep = ' ' if len(row.split(' ')) > 1 else '\t'
            for val in row.split(sep):
                if val[:3] == '/db':
                    fpath = val.strip()
                    args.extn = val.strip().split('.')[-1]
                    break
            wav_id = row.split(sep)[0]
            save_path = "{}/{}.wav".format(args.res_dir, wav_id)  
            if args.extn == 'wav':
                convert_to_wav(fin=fpath, fout=save_path)
            else:
                copyfile(fpath, save_path)
            fpath = save_path
            if isinstance(segments, pd.DataFrame):
                segs = segments[segments[1] == wav_id]
                compress_segments(map_=map_, 
                                  wav_id=wav_id,
                                  file_path=fpath,
                                  segments=segs, 
                                  outpath=args.res_dir)
            else:    
                compress_file(map_=map_, 
                              name=wav_id,
                              save_path=save_path)
            pbar.update(1)


def save_file(map_, args):
    """
    Saves the mapping of utt_id to cr in a file. 
    map_ : mapping of utt_id to cr
    """ 
    if args.segments:
        p = os.path.join(args.res_dir, 'compression_'+args.db+"_seg")
    else:
        p = os.path.join(args.res_dir, 'compression_'+args.db)
    with open(p, 'w') as f:
        for file in map_:
            f.write("{} {}\n".format(file, map_[file]))

def clean_dir(dir):
    """
    Removes all files with certain pattern in the specified directory.
    """
    files = glob.glob(dir)
    for file in files:
        os.remove(file)

def main(args):
    manager = Manager()
    map_ = manager.dict()
    files = ['wav.scp']
    sep = ' '
    if args.db == '6dial':
        sep = '\t'
    for file_ in files: 
        pool = Pool(processes=args.num_process, initargs=(RLock(), ), initializer=tqdm.set_lock)
        processes = []
        csv = open(args.wav_scp, 'r').readlines()
        if args.segments:
             segments = args.wav_scp[:-7] + 'segments'
        else:
            segments = None
        csv_len = len(csv)
        rows_per_process = int(csv_len/args.num_process) + 1
        print('\n')
        print(f"starting processes for file {file_} with {rows_per_process} rows")
        for i in range(args.num_process):
            start = int(i*rows_per_process)
            end = int(min(start + rows_per_process, csv_len))
            processes.append(pool.apply_async(calc_CR_scp, args=(i,
                                map_, 
                                csv, 
                                args, 
                                segments, 
                                start, 
                                end,)))
        pool.close()
        results = [job.get() for job in processes]
   
    for i in range(10):
        print('\n')
    save_file(map_, args)
    print("compression ratio file created successfully...")
    print("cleaning temporary files..")
    clean_dir(dir="/shared/workspaces/anuragkumar95/compressions/*wav*")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help='Type of dataset, cv(commonvoice) or mls/ identifies for compression file')
    parser.add_argument("--num_process", type=int, required=True)
    parser.add_argument('--wav_scp', type=str, required=True)
    parser.add_argument('--res_dir', type=str, required=True, help='Path to dir where the results will be stored.')
    parser.add_argument('--extn', type=str, required=False, help='default audio files extension, if not mentioned sets it internally.')
    parser.add_argument('--segments', action='store_true')
    args = parser.parse_args()
    main(args)


    