import keras
from keras import layers
import numpy as np

path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length : ', len(text))

maxlen = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])
print("sequences : ", len(sentences))

chars = sorted(list(set(text)))

char_indices = dict((char, chars.index(char)) for char in chars)

print('verctorization')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01),
              loss='categorical_crossentropy')
