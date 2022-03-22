from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

num_features = 1000
maxlen = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=num_features)
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(num_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=1, batch_size=128, validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'm*', label="Eğitim Başarimi")
plt.plot(epochs, val_acc, 'g', label="Doğrulama / Geçerleme Başarimi")
plt.legend()

plt.figure()
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'm*', label="Eğitim Kaybi")
plt.plot(epochs, val_acc, 'g', label="Doğrulama / Geçerleme Kaybi")
plt.legend()
