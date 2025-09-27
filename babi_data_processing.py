"""
code adapted from here: https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/babi_memnn.py
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split(r'(\W)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    max_len_s = 0
    max_len_q = 0
    vocab = set()
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            vocab = vocab.union(set(q))
            vocab.add(a)
            max_len_q = max(max_len_q, len(q))
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            vocab = vocab.union(set(sent))
            max_len_s = max(max_len_s, len(sent))
            story.append(sent)
    return data, max_len_s, max_len_q, vocab


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data, max_len_s, max_len_q,vocab = parse_stories(f.readlines(), only_supporting=only_supporting)
    # flatten = lambda data: reduce(lambda x, y: x + y, data)
    # data = [(flatten(story), q, answer) for story, q, answer in data
    #         if not max_length or len(flatten(story)) < max_length]
    return data, max_len_s, max_len_q, vocab


def vectorize_stories(data, max_len_s, max_len_q, word_idx):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        line_tokenized = []
        for line in story:
            line_tokenized.append([word_idx[w] for w in line])

        line_tokenized = pad_sequences(line_tokenized, maxlen=max_len_s)  
        inputs.append(line_tokenized)
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (inputs,
            pad_sequences(queries, maxlen=max_len_q),
            np.array(answers))

try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.tensorflow.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise


challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_'
                                  'single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_'
                                'two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
with tarfile.open(path) as tar:
    train_stories, train_max_len_s, train_max_len_q, train_vocab = get_stories(tar.extractfile(challenge.format('train')))
    test_stories, test_max_len_s, test_max_len_q, test_vocab = get_stories(tar.extractfile(challenge.format('test')))

max_len_s = max(train_max_len_s, test_max_len_s)
max_len_q = max(train_max_len_q, test_max_len_q)
vocab = sorted(train_vocab.union(test_vocab))

# vocab = set()
# for story, q, answer in train_stories + test_stories:
#     vocab |= set(story + q + [answer])
# vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
# story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
# query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, max_len_s, max_len_q, word_idx)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, max_len_s, max_len_q, word_idx)
breakpoint()