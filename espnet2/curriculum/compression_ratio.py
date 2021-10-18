"""
Author: Anurag Kumar
"""
import os
from shutil import copyfile
import pandas as pd
from pydub import AudioSegment
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
    
def compress_segments(map_, file_path, segments, outpath):
    """
    Accepts the wav audio file and list of segment time stamps.
    Chunks the audio and calculates CR for each segment.
    
    file_path : path to wav file
    segments  : list  of segments time stamps
    outpath   : path to save chunks
    """
    audio = AudioSegment.from_wav(file_path)
    filename = file_path.split('/')[-1][:-4]
    for _, row in segments.iterrows():
        start = row[2] * 1000
        end = row[3] * 1000
        audio_chunk = audio[start:end]
        save_path = "{}/{}_chunk_{}_{}.wav".format(outpath, filename, start, end)
        audio_chunk.export(save_path)
        compress_file(map_=map_, 
                      name=row[1],
                      file_path=save_path,
                      save_path=save_path)

def compress_file(map_, name, file_path, save_path):
    """
    Compresses the file and calculates CR.
    """
    size = os.path.getsize(file_path)
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

def calc_CR_scp(pid, map_, file_, type, segments=None, start=None, end=None):
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    files = map_
    data = file_[start:end]
    if segments:
        segments = pd.read_csv(segments, sep = ' ')
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for idx, row in enumerate(data):
            vals = row.split(' ')
            print(vals)
            wav_id = vals[0]
            fpath = vals[8]
            filename = fpath.split('/')[-1]
            fname_out = os.path.join('/shared/workspaces/anuragkumar95/compressions/',filename)
            if type != 'wav':
                convert_to_wav(fin=fpath, fout=fname_out)
                fpath = fname_out
            if segments:
                segs = segments[segments[0] == wav_id]
                compress_segments(map_=map_, 
                                  file_path=fpath, 
                                  segments=segs, 
                                  outpath="/shared/workspaces/anuragkumar95/compressions/")
            else:
                save_path = os.path.join("/shared/workspaces/anuragkumar95/compressions/",filename)
                compress_file(map_=map_, 
                              name=row[0],
                              file_path=fpath,
                              save_path=save_path)
            pbar.update(1)

def calc_CR_MLS(pid, data_dir, map_, file_, start=None, end=None):
    tqdm_text = "#"+"{}".format(pid).zfill(3)
    files = map_
    data = file_[start:end]
    p = os.path.join(data_dir,'audio')
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for idx, row in data.iterrows():
            fname = row['path']
            filename = row['filename']
            fname_in = os.path.join(p, fname)
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
    #p = os.path.join(data_dir,'clips')
    p = data_dir
    files = map_
    data = file_[start:end]
    #for idx, row in tqdm(data.iterrows()): 
    with tqdm(total=end-start, desc=tqdm_text, position=pid+1) as pbar:
        for idx, row in data.iterrows():
            fname = row['path']
            filename = row['filename']
            fname_in = os.path.join(p, fname)
            fname_out = os.path.join('/shared/workspaces/anuragkumar95/compressions/',filename)
            #temp = subprocess.run(["ffmpeg","-i", 
            #                       fname_in, fname_out[:-4]+".wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #temp = subprocess.run(["gzip", "-k", fname_in[:-4]+".wav"])
            temp = subprocess.run(["gzip", "-k", fname_in])
            fsize = os.path.getsize(fname_in)
            fsize_comp = os.path.getsize(fname_in+".gz")
            temp = subprocess.run(["rm", fname_in+".gz"])
            temp = subprocess.run(["rm", fname_in])
            try:
                CR = fsize_comp/fsize
            except Exception as e:
                print(f"File: {fname}, Ori:{fsize}, Compr:{fsize_comp}")
                print(e)
                raise ZeroDivisionError
            files[fname.split('.')[0]] = str(CR)
            pbar.update(1)


def save_file(map_, res_dir, db): 
    p = os.path.join(res_dir, 'compression_'+db)
    with open(p, 'w') as f:
        for file in map_:
            f.write("{} {}\n".format(file, map_[file]))


def main(args):
    manager = Manager()
    map_ = manager.dict()
    if args.db == 'cv':
        files = ['validated.tsv', 'invalidated.tsv']
        sep = '\t'
    if args.db == 'mls':
        files = ['mls_files.tsv']
        sep = '\t'
    else:
        files = ['wav.scp']
        sep = ' '
    for file_ in files: 
        pool = Pool(processes=args.num_process, initargs=(RLock(), ), initializer=tqdm.set_lock)
        processes = []
        csv = pd.read_csv(args.wav_scp, sep = sep)
        print(csv.head())
        if args.segments:
             segments = args.wav_scp[:-7] + 'segments'
        csv_len = len(csv)
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

            else:
                processes.append(pool.apply_async(calc_CR_scp, args=(i,
                                    map_, 
                                    csv, 
                                    args.extn, 
                                    segments, 
                                    start, 
                                    end,)))

        pool.close()
        results = [job.get() for job in processes]
   
    save_file(map_, args.res_dir, args.db)
    print("compression ratio file created successfully...")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help='Type of dataset, cv(commonvoice) or mls')
    parser.add_argument("--num_process", type=int, required=False)
    parser.add_argument('--wav_scp', type=str, required=False)
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')
    parser.add_argument('--extn', type=str, required=True, help='default audio files extension')
    parser.add_argument('--segments', action='store_true')
    args = parser.parse_args()
    main(args)


    