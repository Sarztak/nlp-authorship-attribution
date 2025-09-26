from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras import Sequential
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

docs=["Chuck Berry rolled over everyone who came before him ? and turned up everyone who came after. We'll miss you", "Help protect the progress we've made in helping millions of Americans get covered.", "Let's leave our children and grandchildren a planet that's healthier than the one we have today.", "The American people are waiting for Senate leaders to do their jobs.", "We must take bold steps now ? climate change is already impacting millions of people.", "Don't forget to watch Larry King tonight", "Ivanka is now on Twitter - You can follow her", "Last night Melania and I attended the Skating with the Stars Gala at Wollman Rink in Central Park", "People who have the ability to work should. But with the government happy to send checks", "I will be signing copies of my new book" ]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)
max_len = max([len(s) for s in sequences])
sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

vocab_size = len(tokenizer.word_index) + 1 # 1 is added because index starts at 1 not zero; zero is reserved for padding which is also mapped by the embedding so we need the size of embedding to be N + 1

vec_dim = 8
input_dim = len(docs)

print(f"Vocab Length: {vocab_size}")
print(f"Sequence Length: {max_len}")
print(f"Number of documents: {input_dim}")

# every word is converted to a vector, and the entire document is just a sequence of word vectors

model = Sequential([
    Input(shape=(max_len,)), # the length of the sequence
    Embedding(vocab_size + 1, vec_dim)
])

model.compile('rmsprop', 'mse')
print(model.summary())

output_array = model.predict(sequences)
print(output_array.shape)