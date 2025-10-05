from tensorflow.keras.datasets import reuters
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer 
import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorflow.keras.backend as K
from rich.traceback import install
from wordcloud import WordCloud
import numpy as np 


install()

stopwords  = set(stopwords.words('english'))
word2idx_base = reuters.get_word_index(path="reuters_word_index.json")
rev = {idx + 3: word for word, idx in word2idx_base.items()}


max_words = 5000
idx2word_bow = {idx: rev.get(idx) for idx in range(3, max_words)}

stop_ids_plus3 = {word2idx_base[word] + 3 for word in stopwords if word in word2idx_base}

def remove_stopwords(seqs):
    special_tokens = (0, 1, 2) # 0: padding, 1: start, 2: oov
    """x is a numpy array of python list"""
    for i, seq in enumerate(seqs):
        seqs[i] = [t for t in seq if t not in stop_ids_plus3 and t not in special_tokens]

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1984)

remove_stopwords(x_train)
remove_stopwords(x_val)
remove_stopwords(x_test)

num_classes = max(y_train) + 1

tokenizer = Tokenizer(num_words=max_words)
x_train_bow = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_val_bow = tokenizer.sequences_to_matrix(x_val, mode='binary')
x_test_bow = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
SHUFFLE_BUFFER = 10000

def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(x_train_bow, y_train_one_hot, training=True)
val_ds = make_ds(x_val_bow, y_val_one_hot)
test_ds = make_ds(x_test_bow, y_test_one_hot)

class MLPAttention(Model):
    def __init__(self, max_words, num_classes):
        super().__init__()
        self.attention = layers.Dense(max_words, activation='softmax', name='attention')
        self.mult = layers.Multiply(name='attention_prod')
        self.ff = layers.Dense(256, activation='relu', name='ff')
        self.out = layers.Dense(num_classes, activation='softmax', name='classifier')
        

    def call(self, x, return_intermediate=False):
        a  = self.attention(x)
        xw = self.mult([x, a])
        h  = self.ff(xw)
        y  = self.out(h)
        if return_intermediate:
            return {"attention": a, "attention_prod": xw, "ff": h, "output": y}
        return y
    
def get_activation_layer(model, x, layer_name):
    sub = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return sub.predict(x)


if __name__ == "__main__":
    model = MLPAttention(max_words, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=5)

    x, _ = next(iter(train_ds))
    acts = model(x, return_intermediate=True)
    attn = acts["attention"].numpy()
    
    
    present_weights = attn * x.numpy()
    scores = present_weights[0] # batch 0
    k = min(100, int((scores > 0).sum()))
    topk_idx = np.argsort(scores)[-k:].tolist()
    freqs = {idx2word_bow[i]: float(scores[i]) for i in topk_idx if i in idx2word_bow}


    wc = WordCloud(width=800, height=400, background_color="white")
    img = wc.generate_from_frequencies(freqs).to_image()
    img.show()
