import sentencepiece as spm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import argparse


def train_vector_model(subword_model, text, save_file, sep='\t'):
    '''
    Params:
        subword_model (str): Path to sentencepiece model.
        text (str) path to text in scp format.
        sep (str) separator between sentence ID and sentence.
        save_file (str) file name to store vectors.
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
    print(f"Saved vectors to {save_file}.")
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
    print(f"Saved word norms to {save_file}.")


def calc_sent_norm_complexity(word_norms_file, text, save_file):
    data_dict = {}
    print("Reading text data...")
    with open(text, 'r') as fo:
        for line in fo.readlines():
            data_dict[line.split(sep)[0]] = line.split(sep)[-1].strip()
    
    word_norms = {}
    print("Reading word norms...")
    with open(word_norms_file, 'r') as fo:
        for line in fo.readlines():
            word_norms[line.split()[0]] = float(line.split()[-1])

    print("Calculating sentence norms...")
    sent_norms = {}

    for k in tqdm(data_dict):
        sent_norm = 0
        sent = data_dict[k].split()
        for w in sent:
            sent_norm+=word_norms[w]
        sent_norm/=len(sent)
        sent_norms[k]=sent_norm
    
    with open(save_file, 'w') as fo:
        for k in sent_norms:
            fo.write(k+' '+str(sent_norms[k])+'\n')
    print(f"Saved sentence norms in {save_file}.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,required=True, help='vectors, wnorms or snorms.')
    parser.add_argument('--text', type=str, help='Path to text file in SCP format.')
    parser.add_argument('--subword_model', type=str, help='Path to sentencepiece model.')
    parser.add_argument('--sep', default='\t', type=str, help='Separator between sentence ID and sentence.')
    parser.add_argument('--save_file', type=str, help='File to save the result of the function.')
    parser.add_argument('--vectors', type=str, help='Path to file with saved vectors.')
    parser.add_argument('--word_norms', type=str, help='Path to file with precalculated word norms.')

    args = parser.parse_args()

    if args.task=='vectors':
        train_vector_model(args.subword_model, 
                           args.text, 
                           args.save_file,
                           args.sep)

    if args.task=='wnorms':
        calculate_word_norms(args.vectors_file,
                             args.subword_model, 
                             args.text, 
                             args.save_file)

    if args.task='snorms':
        calc_sent_norm_complexity(args.word_norms_file, 
                                  args.text, 
                                  args.save_file)
