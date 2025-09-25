from dotenv import load_dotenv
load_dotenv() # suppress tensorflow warning messages

import re 
import os 
from pathlib import Path
import numpy as np 
from rich.traceback import install
import tensorflow as tf
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten, Input
from nltk.util import ngrams    
from sklearn.model_selection import train_test_split

install()

def segment_document_words(filename, words_per_segment):
    word_dict = {}
    words = []
    for line in open(filename, 'r', encoding='latin-1'):
        tokens = line.rstrip().split(" ")
        for token in tokens:
            if token != "":
                words.append(token)
                word_dict[token] = 1

    segments = [words[i:i + words_per_segment] for i in range(0, len(words), words_per_segment)]
    
    return segments, word_dict

def create_label(filename):
    pattern = re.compile(".+([A-Za-z])[0-9]\.(txt|TXT)$")
    match = re.match(pattern, filename)
    return match.group(1) if match else None

def create_label_dict(train_dir_path):
    label_dict = dict()
    for filename in os.listdir(train_dir_path):
        label = create_label(filename)
        if label:
            label_dict[ord(label) - 65] = label

    return label_dict 


def vectorize_documents_bow(data_path, words_per_segment, labels_dict):
    segments = []
    vocab = {}
    labels = []
    x, y = [], []

    for file in os.listdir(data_path):
        if not (label:=create_label(file)):
            continue
        segmented_doc, vocab_dict = segment_document_words(Path(data_path) / file, words_per_segment)
        vocab.update(vocab_dict)
        segments.extend(segmented_doc)
        n_label = ord(label) - 65
        labels.extend([n_label for _ in segmented_doc])

    for segment in segments:
        segment = ' '.join(segment)
        x.append(pad_sequences([hashing_trick(segment, round(len(vocab)*1.3))], words_per_segment)[0])

    y = to_categorical(labels, len(labels_dict))
    
    return np.array(x), y, len(vocab)

def segment_document_ngrams(filename, words_per_segment, ngram_size):
    word_dict = {}
    words = []

    for line in open(filename, 'r', encoding='latin-1'):
        ngram_list = ngrams(line.rstrip().split(" "), ngram_size)
        for ngram in ngram_list:
            joined = "_".join(ngram)
            words.append(joined)
            word_dict[joined] = 1

    segments = [words[i:i + words_per_segment] for i in range(0, len(words), words_per_segment)]
    
    return segments, word_dict
    
def vectorize_documents_ngram(data_path, words_per_segment, labels_dict, ngram_size):
    segments = []
    vocab = {}
    label_num = list(labels_dict.keys())
    label = list(labels_dict.values())
    x, y = [], []

    for file in os.listdir(data_path):
        if file == "README.txt":
            continue
        print(file)
        segmented_doc, vocab_dict = segment_document_ngrams(Path(data_path) / file, words_per_segment, ngram_size)
        vocab.update(vocab_dict)
        segments.extend(segmented_doc)

    for segment in segments:
        segment = ' '.join(segment)
        x.append(pad_sequences([hashing_trick(segment, round(len(vocab)*1.3))], words_per_segment)[0])
        y = to_categorical(label_num, len(label_num))
    
    return np.array(x), y, vocab

def segment_document_char_ngrams(filename, words_per_segment, ngram_size):
    word_dict = {}
    words = []

    for line in open(filename, 'r', encoding='latin-1'):
        line = line.rstrip().replace(" ", "#")
        char_ngrams_list = ngrams(list(line), ngram_size)
        for char_ngram in char_ngrams_list:
            joined = "".join(char_ngram)
            words.append(joined)
            word_dict[joined] = 1

    segments = [words[i:i + words_per_segment] for i in range(0, len(words), words_per_segment)]
    
    return segments, word_dict


    
def vectorize_documents_char_ngram(data_path, words_per_segment, labels_dict, ngram_size):
    segments = []
    vocab = {}
    label_num = list(labels_dict.keys())
    label = list(labels_dict.values())
    x, y = [], []

    for file in os.listdir(data_path):
        if file == "README.txt":
            continue
        print(file)
        segmented_doc, vocab_dict = segment_document_char_ngrams(Path(data_path) / file, words_per_segment, ngram_size)
        vocab.update(vocab_dict)
        segments.extend(segmented_doc)

    for segment in segments:
        segment = ' '.join(segment)
        x.append(pad_sequences([hashing_trick(segment, round(len(vocab)*1.3))], words_per_segment)[0])
        y = to_categorical(label_num, len(label_num))
    
    return np.array(x), y, vocab


def model(vocab_len, input_dim, output_dim, nb_classes):
    m = Sequential([
        Input(shape=(input_dim,)),
        Embedding(vocab_len, output_dim),
        Flatten(),
        Dense(output_dim, activation="relu"),
        Dropout(0.3),
        Dense(nb_classes, activation="sigmoid")
    ])
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    return m

if __name__ == "__main__":
    path = r"pan12-authorship-attribution-corpora\pan12-authorship-attribution-training-corpus-2012-03-28\pan12-authorship-attribution-training-corpus-2012-03-28"
    labels_dict = create_label_dict(path)
    nb_classes = len(labels_dict)

    X, y, vocab_len = vectorize_documents_bow(path, 20, labels_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    input_dim = 20
    output_dim = 300
    model = model(round(vocab_len*1.3), input_dim, output_dim, nb_classes)
    print(model.summary())    

    nb_epochs = 10

    model.fit(X_train, y_train, epochs=nb_epochs, shuffle=True, batch_size=64, validation_split=0.3, verbose=2)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f'Accuracy: {accuracy*100} % ')