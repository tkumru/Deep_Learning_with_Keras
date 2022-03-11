from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Classification of MNIST dataset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(f"Images dimension: {train_images.ndim}")
print(f"Images shape (numbers, width, length): {train_images.shape}")
print()
# Neural Network Architecture

network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28*28,)))
network.add(Dense(10, activation='softmax'))

# Optimize Neural Network

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Preproccessing Dataset

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Training Neural Network

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}")
