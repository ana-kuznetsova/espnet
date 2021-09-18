import os
import sys
import pandas as pd
from string import punctuation
punctuation+='—'
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def filter_wav(wav, map, save_dir):
    save_dir = os.path.join(save_dir, 'sentence_norm.txt')
    with open(save_dir, 'w') as fo:
        for line in tqdm(open(wav,'r').readlines()):
            _id_ = line.split(' ')[0]
            if _id_ in map:
                fo.write(_id_+" "+map[_id_]+'\n')
            else:
                print(_id_)
    print("sentence length written successfully")

def main(lang, corpus_path, cv_path, save_dir):

    #Open Covo corpora
    with open(os.path.join(corpus_path, lang+'.txt'), 'r') as fo:
        f = fo.readlines()

    #Remove punctuation from covo corpus
    filtered_text = [''.join([char for char in line.lower() if char not in punctuation]) for line in f]

    #Collect all the sentences from  covo and CV files into one dict
    sent_dict = {}

    for i, line in enumerate(filtered_text):
        sent_dict[i]=line

    val = pd.read_csv(os.path.join(cv_path,'validated.tsv'), sep='\t')
    inval = pd.read_csv(os.path.join(cv_path,'invalidated.tsv'), sep='\t')
    df = pd.concat([val, inval])

    df_txt = []

    for line in df['sentence']:
        df_txt.append(''.join([char for char in line.lower() if char not in punctuation]))

    for i, line in enumerate(df['path']):
        sent_dict[line] = df_txt[i]

    #Make spm file
    with open(os.path.join(corpus_path, lang+'_spm'), 'w') as fo:
        for k in sent_dict:
            fo.write(sent_dict[k]+'\n')
    
    #Train sentencepiece model
    print("SPM training started...")
    spm_path = os.path.join(corpus_path, lang+"_spm")
    cmd = '--input='+spm_path+' --model_prefix='+os.path.join(corpus_path, lang+'_unigram ')+ '--vocab_size=8000 ' + ' --character_coverage=1'
    spm.SentencePieceTrainer.train(cmd)

    #Train Word2Vec model

    subword_model = os.path.join(corpus_path, lang+'_unigram'+'.model')
    sp = spm.SentencePieceProcessor()
    sp.Load(subword_model)

    print("Creating subwords for training data...")
    training_data = []

    for k in tqdm(sent_dict):
        enc_s = sp.EncodeAsPieces(sent_dict[k])
        training_data.append(enc_s)

    print("Training subword vectors...")
    model = Word2Vec(sentences=training_data, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    word_vectors.save(os.path.join(corpus_path, lang+'.wordvectors'))
    print("Saved vectors to ", os.path.join(corpus_path, lang+'.wordvectors'))

    print("Creating scp files...")
    scp_text = []

    for i, row in val.iterrows():
        ID = row['client_id']+'-'+row['path'].replace('.mp3', '')
        txt = row['sentence']
        txt = ''.join([char for char in txt.lower() if char not in punctuation])
        scp_text.append(ID+'\t'+txt+'\n')

    for i, row in inval.iterrows():
        ID = row['client_id']+'-'+row['path'].replace('.mp3', '')
        txt = row['sentence']
        txt = ''.join([char for char in txt.lower() if char not in punctuation])
        scp_text.append(ID+'\t'+txt+'\n')

    with open(os.path.join(corpus_path, lang+'_text.scp'), 'w') as fo:
        for line in scp_text:
            fo.write(line+'\n')

    print("Calculating max subword norm...")
    max_norm = 0

    for line in tqdm(scp_text):
        for sub in line.split('\t')[-1].strip():
            if sub in word_vectors:
                max_norm = max(max_norm, np.linalg.norm(word_vectors[sub]))


    word_norms = {}

    print("Calculating word norms...")
    for line in tqdm(scp_text):
        sent = line.split('\t')[-1]
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
            if sub_word in word_vectors:
                norm += np.linalg.norm(word_vectors[sub_word])
            else:
                norm+=max_norm+1

    max_word_norm = max(word_norms.items(), key = lambda x: x[1])[1]

    print("Calculating sentence norms...")
    sent_norms = {}

    for line in tqdm(scp_text):
        sent_norm = 0
        k = line.split('\t')[0]
        sent = line.split('\t')[-1]
        for w in sent:
            if w in word_norms:
                sent_norm+=word_norms[w]
            else:
                sent_norm+=max_word_norm
        sent_norm/=len(sent)
        sent_norms[k]=sent_norm

    
    filter_wav()


if __name__=="__main__":
    lang = sys.argv[1]
    corpus_path = sys.argv[2]
    cv_path = sys.argv[3]
    save_dir = sys.argv[4]
    main(lang, corpus_path, cv_path, save_dir)
