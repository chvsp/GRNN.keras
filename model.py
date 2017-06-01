import keras
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding

from grounded_layers import Grounded_GRU, Grounded_Dense


def GRNN(hidden_size, num_labels, num_words=100, embed_length=32, max_input_length=50):

    model = Sequential()
    model.add(Embedding(num_words, embed_length, input_length=max_input_length))
    model.add(Grounded_GRU(hidden_size, num_labels))
    model.add(Grounded_Dense(num_labels, activation='sigmoid'))

    return model

if __name__ == '__main__':
    
    model = GRNN(32, 10)

    print model.summary()