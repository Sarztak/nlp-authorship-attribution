from tensorflow.keras.datasets import reuters
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer 
import tensorflow as tf

stopwords  = set(stopwords.words('english'))

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1984)

word_index = reuters.get_word_index(path="reuters_word_index.json")
word_index = {idx: word for word, idx in word_index.items()}

max_words = 10000
num_classes = max(y_train) + 1

tokenizer = Tokenizer(num_words=max_words)
x_train_bow = tokenizer.sequence_to_matrix(x_train, mode='binary')
x_val_bow = tokenizer.sequence_to_matrix(x_val, mode='binary')
x_test_bow = tokenizer.sequence_to_matrix(x_test, mode='binary')

y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)





breakpoint()