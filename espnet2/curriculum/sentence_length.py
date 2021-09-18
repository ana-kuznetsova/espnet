import os
import pathlib 
import argparse
import pandas as pd
from tqdm import tqdm

def parse_csv(csv, map):
    df = pd.read_csv(csv, sep = '\t')
    for _id_, path, sent in zip(df['client_id'], df['path'], df['sentence']):
        key = _id_+"-"+pathlib.Path(path).stem
        map[key] = str(len(sent))

def filter_wav(wav, map, save_dir):
    save_dir = os.path.join(save_dir, 'sentence_length.txt')
    with open(save_dir, 'w') as fo:
        for line in tqdm(open(wav,'r').readlines()):
            _id_ = line.split(' ')[0]
            if _id_ in map:
                fo.write(_id_+" "+map[_id_]+'\n')
            else:
                print(_id_)
    print("sentence length written successfully")

def main(args):
    map = {}
    for file in ['validated.tsv', 'invalidated.tsv']:
        fpath = os.path.join(args.data_dir, file)
        parse_csv(fpath, map)
    filter_wav(args.wav_scp, map, args.res_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_scp', type=str, required=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')
    args = parser.parse_args()
    main(args)