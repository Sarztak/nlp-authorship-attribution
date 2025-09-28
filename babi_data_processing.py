"""
dataset extracting partially adapted from here: https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/babi_memnn.py
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tarfile
import numpy as np
import re
from rich.traceback import install 

install()

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
    max_sent = 0
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
            max_sent = max(max_sent, len(substory))
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            vocab = vocab.union(set(sent))
            max_len_s = max(max_len_s, len(sent))
            story.append(sent)
    return data, max_len_s, max_len_q, max_sent, vocab


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data, max_len_s, max_len_q, max_sent, vocab = parse_stories(f.readlines(), only_supporting=only_supporting)
    return data, max_len_s, max_len_q, max_sent, vocab


def vectorize_stories(data, max_len_s, max_len_q, max_sent, word_idx):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        line_tokenized = []
        for line in story:
            line_tokenized.append([word_idx[w] for w in line])

        line_tokenized = pad_sequences(line_tokenized, maxlen=max_len_s)  
        inputs.append(line_tokenized)
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])

    inputs_arr = np.zeros(shape=(len(inputs), max_sent, max_len_s))
    
    for i in range(len(inputs)):
        pad_x = max_sent - inputs[i].shape[0]
        sent_padding = np.zeros(shape=(pad_x, max_len_s))
        padded_story = np.concat([inputs[i], sent_padding], axis=0)
        inputs_arr[i] = padded_story

    return (inputs_arr,
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
    train_stories, train_max_len_s, train_max_len_q, train_max_sent, train_vocab = get_stories(tar.extractfile(challenge.format('train')))
    test_stories, test_max_len_s, test_max_len_q, test_max_sent, test_vocab = get_stories(tar.extractfile(challenge.format('test')))

max_len_s = max(train_max_len_s, test_max_len_s)
max_len_q = max(train_max_len_q, test_max_len_q)
max_sent = max(train_max_sent, test_max_sent)
vocab = sorted(train_vocab.union(test_vocab))



# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, max_len_s, max_len_q, max_sent, word_idx)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, max_len_s, max_len_q, max_sent, word_idx)

train_ds =  tf.data.Dataset.from_tensor_slices(((inputs_train, queries_train), tf.one_hot(answers_train, vocab_size)))

test_ds =  tf.data.Dataset.from_tensor_slices(((inputs_test, queries_test), tf.one_hot(answers_test, vocab_size)))

BATCH_SIZE =  32
BUFFER_SIZE = 10000
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)
# ----------------------------------------------
# MODEL
# ----------------------------------------------

class MemN2N(Model):
    def __init__(self, max_sent, max_len_q, max_len_s, vocab_size, emb_dim, num_hops):
        super().__init__()
        self.max_sent = max_sent # sentences per story (padded)
        self.max_len_q = max_len_q # words per question (padded)
        self.max_len_s = max_len_s # words per sentence (padded)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_hops = num_hops

        self.PE_s = self.position_encoding(self.max_len_s, self.emb_dim)
        self.PE_q = self.position_encoding(self.max_len_q, self.emb_dim)
        
        self.key_emb = Embedding(self.vocab_size, self.emb_dim)
        self.query_emb = Embedding(self.vocab_size, self.emb_dim)
        self.value_emb = Embedding(self.vocab_size, self.emb_dim)
        self.flatten = Flatten()
        self.dense = Dense(self.vocab_size)
    
    def position_encoding(self, L, d):
        # L: sentence length, d: embedding dim
        j = tf.range(1, L+1, dtype=tf.float32)[:, None]  # (L,1)
        k = tf.range(1, d+1, dtype=tf.float32)[None, :]  # (1,d)
        L_jk = (1.0 - j / L) - (k / d) * (1.0 - 2.0 * j / L)  # (L,d)
        return L_jk

    def call(self, x):
        story, que = x
        key = self.key_emb(story)
        value = self.value_emb(story)
        query = self.query_emb(que)

        key = tf.einsum('bsld,ld->bsd', key, self.PE_s)
        value = tf.einsum('bsld,ld->bsd', value, self.PE_s)
        query = tf.einsum('bqd,qd->bd', query, self.PE_q)

        for _ in range(self.num_hops):
            scores = tf.einsum('bd,bsd->bs', query, key)
            p = tf.nn.softmax(scores, axis=-1)
            o = tf.einsum('bs,bsd->bd', p, value)

            query = query + o

        logits = self.dense(query)

        return logits


model = MemN2N(max_sent, max_len_q, max_len_s, 
               vocab_size, emb_dim=64, num_hops=2)



model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, epochs=5)

loss, accuracy = model.evaluate(test_ds)
print(accuracy)