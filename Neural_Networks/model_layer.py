from keras import models
from keras import layers

# MODEL --------------------------

model = models.Sequential()

# LAYERS -------------------------

model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32))

print()
model.summary()
print()
