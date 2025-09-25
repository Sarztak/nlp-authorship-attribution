from dotenv import load_dotenv
load_dotenv() # suppress loading tensorflow warning messages

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
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten, Input, Convolution1D, LSTM, Lambda
from nltk.util import ngrams    
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from collections import defaultdict

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
    pattern = re.compile(".+([A-Za-z])[0-9]\\.(txt|TXT)$")
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
    labels = []
    x, y = [], []

    for file in os.listdir(data_path):
        if not (label:=create_label(file)):
            continue
        segmented_doc, vocab_dict =  segment_document_ngrams(Path(data_path) / file, words_per_segment, ngram_size)
        vocab.update(vocab_dict)
        segments.extend(segmented_doc)
        n_label = ord(label) - 65
        labels.extend([n_label for _ in segmented_doc])

    for segment in segments:
        segment = ' '.join(segment)
        x.append(pad_sequences([hashing_trick(segment, round(len(vocab)*1.3))], words_per_segment)[0])
        y = to_categorical(labels, len(labels_dict))
    
    return np.array(x), y, len(vocab)

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
    labels = []
    x, y = [], []

    for file in os.listdir(data_path):
        if not (label:=create_label(file)):
            continue
        segmented_doc, vocab_dict = segment_document_char_ngrams(Path(data_path) / file, words_per_segment, ngram_size)
        vocab.update(vocab_dict)
        segments.extend(segmented_doc)
        n_label = ord(label) - 65
        labels.extend([n_label for _ in segmented_doc])

    for segment in segments:
        segment = ' '.join(segment)
        x.append(pad_sequences([hashing_trick(segment, round(len(vocab)*1.3))], words_per_segment)[0])
        y = to_categorical(label_num, len(label_num))
    
    return np.array(x), y, len(vocab)


def model(vocab_len, input_dim, output_dim, nb_classes):
    m = Sequential([
        Input(shape=(input_dim,)),
        Embedding(vocab_len, output_dim),
        Dense(output_dim, activation="relu"),
        Dropout(0.3),
        Flatten(),
        Dense(nb_classes, activation="sigmoid")
    ])
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    return m

def cnn_model(vocab_len, input_dim, output_dim, nb_classes):
    m = Sequential([
        Input(shape=(input_dim,)),
        Embedding(vocab_len, output_dim),
        Dense(output_dim, activation="relu"),
        Convolution1D(32, 30, padding="same"),
        Flatten(),
        Dense(nb_classes, activation="sigmoid")
    ])
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    return m


def get_random_samples(texts, max_samples_per_author):
    nb_texts = len(texts)
    nb_samples = min(max_samples_per_author, nb_texts)
    docs = np.array(texts)
    random_indexes = np.random.choice(docs.shape[0], nb_samples, replace=False)
    samples = docs[random_indexes]
    return samples

def exp_neg_manhattan_distance(x, y):
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

def split_data(X, y, max_samples_per_author=10):
    X, y = shuffle(X, y, random_state=1984)
    authors_x = defaultdict(list)

    for x, y in zip(X, y):
        author_id = np.where(y==1)[0][0]
        authors_x[author_id].append(x)
    
    X_left, X_right, y_lr = [], [], []
    done = set()

    for author, texts in authors_x.items():
        left_samples = get_random_samples(texts, max_samples_per_author)
        for other_author, other_texts in authors_x.items():
            if (other_author, author) in done:
                continue
            done.add((author, other_author))
            right_samples = get_random_samples(other_texts, max_samples_per_author)
        
            for l, r in zip(left_samples, right_samples):
                X_left.append(l)
                X_right.append(r)
                y_lr.append(float(author == other_author))
    
    return np.array(X_left), np.array(X_right), np.array(y_lr)            


def siamese_network(input_dim, output_dim, vocab_len, nb_units=10):
    left_input = Input(shape=(input_dim,))
    right_input = Input(shape=(input_dim,))
    embedding =  Embedding(vocab_len, output_dim, input_length=input_dim)
    encoded_left = embedding(left_input)
    encoded_right = embedding(right_input)
    lstm = LSTM(nb_units)
    left_output = lstm(encoded_left)
    right_output = lstm(encoded_right)

    model_distance = Lambda(function=lambda x: exp_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    model = Model([left_input, right_input], [model_distance])
    
    return model


def run_auth_attribution(model, X_train, X_test, y_train, y_test, nb_epochs):
    print(model.summary())
    model.fit(X_train, y_train, epochs=nb_epochs, shuffle=True, batch_size=64, validation_split=0.3, verbose=2)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy: {accuracy*100} % ')

def run_auth_verification(model, X_train, X_test, y_train, y_test, nb_epochs, max_samples_per_author=20):
    X_train_left, X_train_right, y_train_lr = split_data(X_train, y_train, max_samples_per_author)
    X_test_left, X_test_right, y_test_lr = split_data(X_test, y_test, max_samples_per_author)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["acc"])
    print(model.summary())
    model.fit([X_train_left, X_train_right], y_train_lr, epochs=nb_epochs, shuffle=True, batch_size=64, validation_split=0.3, verbose=2)
    loss, accuracy = model.evaluate([X_test_left, X_test_right], y_test_lr, verbose=0)
    print(f'Accuracy: {accuracy*100} % ')

if __name__ == "__main__":
    drive_path = Path("/content/drive/MyDrive/pan12_auth_att/pan12-authorship-attribution-corpora/pan12-authorship-attribution-corpora")
    path = drive_path / "pan12-authorship-attribution-training-corpus-2012-03-28/pan12-authorship-attribution-training-corpus-2012-03-28"

    labels_dict = create_label_dict(path)
    nb_classes = len(labels_dict)

    input_dim = 500
    output_dim = 300
    nb_epochs = 10
    
    X, y, vocab_len = vectorize_documents_bow(path, input_dim, labels_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1984)

    # model = cnn_model(round(vocab_len*1.3), input_dim, output_dim, nb_classes)
    siamese_model = siamese_network(input_dim, output_dim, vocab_len)
    run_auth_verification(siamese_model, X_train, X_test, y_train, y_test, nb_epochs, 20)    
