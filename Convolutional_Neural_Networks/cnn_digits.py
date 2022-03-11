from tabnanny import verbose
from keras.datasets import mnist
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
plt.figure(figsize=(14, 14))
x, y = 10, 4

for i in range(40):
    plt.subplot(y, x, i + 1)
    plt.imshow(x_train[i])

plt.show()
"""

batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cls = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cls, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cls, 1)
    input_shape = (img_rows, img_cls, 1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))

print("*"*100)
print(model.summary())
print("*"*100)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose)
print()
print()
print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])
