import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())

vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1
vocab_size = len(vocab)

inverse_vocab = {index: token for token, index in vocab.items()}

example_sequence = [vocab[word] for word in tokens]

window_size = 2
positive_skip_grams, _ = tf.compat.v2.keras.preprocessing.sequence.skipgrams(
      example_sequence,
      vocabulary_size=int(vocab_size),
      window_size=int(window_size),
      negative_samples=1.0)

print(positive_skip_grams)