import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Input
import numpy as np 


seq_len = 10
vocab_size = 100 #
vec_dim = 8
model = Sequential([
    Input(shape=(seq_len, )),
    Embedding(vocab_size, vec_dim)
])

input_array = np.random.randint(100, size=(10, 10))

model.compile('rmsprop', 'mse')
print(model.summary())

output_array = model.predict(input_array) # output_array.shape = (10, 10, 8)

