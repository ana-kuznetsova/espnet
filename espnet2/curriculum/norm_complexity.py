import sentencepiece as spm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm


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