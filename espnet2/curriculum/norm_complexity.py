#!/usr/bin/python
# coding:utf-8
# Author : Anastasiia Kuznetsova

import sentencepiece as spm
from string import punctuation
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd

'''
The scirpt calculates vector norm as a complexity measure for curriculum learning.

The word norms are calculating using Word2Vec model trained on subwords generated by 
Sentencepiece model. The sentence norm is an average of word norms.

For argument information run:

python norm_complexity.py --help
'''

def pre_process(text, lang=None, corpus='CV'):
    '''
    Params:
        text(str): path to raw text.
        lang (str) language.
        cv_path (str) path to commonvoice root folder.
    '''
    if corpus=='CV':
        CV_ROOT = '~/Rev/commonvoice'

    raw_text = [line.lower() for line in open(text, 'r').readlines()]
    
    if lang:
        csv = pd.read_csv(CV_ROOT+'/'+lang+'/validated.tsv')
        for id_, row in csv.iterrows(): 
            raw_text.append(row['sentence'].lower())

    filtered_text = [''.join([char for char in line if char not in punctuation]) for line in raw_text]
    
    with open(text.split('.')[0]+'_filtered.txt', 'w') as fo:
        for i, line in enumerate(filtered_text):
            fo.write(str(i)+' '+line)

def train_vector_model(subword_model, text, save_file, lang=None, sep='\t'):
    '''
    Params:
        subword_model (str): Path to sentencepiece model.
        text (str) path to text in scp format.
        sep (str) separator between sentence ID and sentence.
        save_file (str) file name to store vectors.
        lang (str) langauge
    '''
    sp = spm.SentencePieceProcessor()
    sp.Load(subword_model)

    data_dict = {}
    print("Reading text data...")
    with open(text, 'r') as fo:
        for line in fo.readlines():
            data_dict[line.split(sep)[0]] = line.split(sep)[-1].strip()

    print("Creating subwords for training data...")
    training_data = []

    for k in tqdm(data_dict):
        enc_s = sp.EncodeAsPieces(data_dict[k])
        training_data.append(enc_s)

    print("Training subword vectors...")
    model = Word2Vec(sentences=training_data, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    word_vectors.save(save_file)
    print("Saved vectors to ", save_file)
    return word_vectors, training_data


def calculate_word_norms(vectors_file, subword_model, text, save_file):
    sp = spm.SentencePieceProcessor()
    sp.Load(subword_model)

    wv = KeyedVectors.load(vectors_file, mmap='r')

    data_dict = {}
    print("Reading text data...")
    with open(text, 'r') as fo:
        for line in fo.readlines():
            data_dict[line.split(sep)[0]] = line.split(sep)[-1].strip()

    word_norms = {}

    for k in tqdm(data_dict):
        sent = data_dict[k]
        sent = sp.EncodeAsPieces(sent) + ['▁']
        token = ''
        norm = 0
        for j, sub_word in enumerate(sent):
            if '▁' in sub_word and len(token) > 0:
                token = token.replace('▁', '')
                word_norms[token] = norm
                token = ''
                token+=sub_word
                norm = 0
                norm += np.linalg.norm(wv[sub_word])
            else:
                token += sub_word
                norm += np.linalg.norm(wv[sub_word]) 
    
    with open(save_file, 'w') as fo:
        for k in word_norms:
            fo.write(k+' '+str(word_norms[k])+'\n')
    print("Saved word norms to ", save_file)


def calc_sent_norm_complexity(vectors_file, subword_model, text, save_file, max_norm, sep):
    sp = spm.SentencePieceProcessor()
    sp.Load(subword_model)

    wv = KeyedVectors.load(vectors_file, mmap='r')
    data_dict = {}

    print("Reading text data...")
    with open(text, 'r') as fo:
        for line in fo.readlines():
            data_dict[line.split(sep)[0]] = line.split(sep)[-1].strip()
    
    word_norms = {}

    print("Calculating word norms...")
    for k in tqdm(data_dict):
        sent = data_dict[k]
        sent = sp.EncodeAsPieces(sent) + ['▁']
        token = ''
        norm = 0
        for j, sub_word in enumerate(sent):
            if '▁' in sub_word and len(token) > 0:
                token = token.replace('▁', '')
                word_norms[token] = norm
                token = ''
                token+=sub_word
                norm = 0
            else:
                token += sub_word
            if sub_word in wv:
                norm += np.linalg.norm(wv[sub_word])
            else:
                norm+=max_norm+1

    print("Calculating sentence norms...")
    sent_norms = {}

    for k in tqdm(data_dict):
        sent_norm = 0
        sent = data_dict[k].split()
        for w in sent:
            if w in word_norms:
                sent_norm+=word_norms[w]
            else:
                sent_norm+=max_norm
        sent_norm/=len(sent)
        sent_norms[k]=sent_norm
    
    with open(save_file, 'w') as fo:
        for k in sent_norms:
            fo.write(k+' '+str(sent_norms[k])+'\n')
    print("Saved sentence norms in ", save_file)

def main(args):
    #pre_process(args.text, args.lang)
    #filtered_text = args.text.split('.')[0] + '_filtered.txt'
    if args.task=='vectors':
        train_vector_model(args.subword_model, 
                           args.text, 
                           args.save_file,
                           args.sep)

    elif args.task=='wnorms':
        calculate_word_norms(args.vectors_file,
                             args.subword_model, 
                             args.text, 
                             args.save_file)
    else:
        calc_sent_norm_complexity(args.vectors,
                                  args.subword_model, 
                                  args.text, 
                                  args.save_file,
                                  args.max_norm,
                                  args.sep)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,required=True, help='vectors, wnorms or snorms.')
    parser.add_argument('--text', type=str, help='Path to text file in SCP format.')
    parser.add_argument('--subword_model', type=str, help='Path to sentencepiece model.')
    parser.add_argument('--sep', default='\t', type=str, help='Separator between sentence ID and sentence.')
    parser.add_argument('--save_file', type=str, help='File to save the result of the function.')
    parser.add_argument('--vectors', type=str, help='Path to file with saved vectors.')
    parser.add_argument('--word_norms', type=str, help='Path to file with precalculated word norms.')
    parser.add_argument('--max_norm', default=999999999999.0, type=float, help='Max norm for filling OOVs')
    parser.add_argument('--corpus', type=str, required=False)
    parser.add_argument('--lang', type=str, required=False)
    args = parser.parse_args()
    main(args)