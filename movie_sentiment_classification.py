import os 
import re 
import shutil 
import string 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from rich.traceback import install 
from pathlib import Path
from tensorflow.keras import layers, losses, utils

install()

# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = utils.get_file("aclImdb_v1", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')

dataset_dir = Path("./aclImdb_v1/aclImdb")
train_dir = dataset_dir / "train"
test_dir = dataset_dir / "test"

batch_size = 32
seed = 1984

raw_train_ds = utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2, subset="training",
    seed=seed
)
raw_val_ds = utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2, subset="validation",
    seed=seed
)

raw_test_ds = utils.text_dataset_from_directory(
    test_dir, batch_size=batch_size
)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, f"{re.escape(string.punctuation)}", '')


max_features = 10000 # fixed vocabulary size
sequence_length = 250 # fixed number of tokens per review

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_sequence_length=sequence_length
) 

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1) # batch dimension is added
    return vectorize_layer(text), label 


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_train_ds.map(vectorize_text)
test_ds = raw_train_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(loss=losses.BinaryCrossentropy(), optimizer='adam', metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

loss, accuracy = model.evaluate(test_ds)

print(f"Loss: {loss}; Accuracy: {accuracy}")

history_dict = history.history
acc = history_dict.get("binary_accuracy")
val_acc = history_dict.get("val_binary_accurary")
loss = history.get("loss")
val_loss = history.get("val_loss")

breakpoint()

epochs = range(1, len(epochs) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 8))

ax[1, 1].plot(epochs, loss, 'bo', label="Training Loss")
ax[1, 1].plot(epochs, val_loss, 'r-', label="Validation Loss")
ax[1, 1].xlabel('Epochs')
ax[1, 1].ylabel('Loss')
ax[1, 1].legend()

ax[1, 2].plot(epochs, acc, 'bo', label="Training Accuracy")
ax[1, 2].plot(epochs, val_acc, 'r-', label="Validation Accuracy")
ax[1, 2].xlabel('Epochs')
ax[1, 2].ylabel('Loss')
ax[1, 2].legend()

plt.show()
