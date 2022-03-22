import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

model = VGG16(weights="imagenet", include_top=True)
layers = dict([(layer.name, layer.output) for layer in model.layers])

image_path = 'Sample_Applications/image/human.jpeg'
image = Image.open(image_path)
image = image.resize((224, 224))

x = np.array(image, dtype="float32")
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction = model.predict(x)
print("Predicted: ", decode_predictions(prediction, top=3)[0])
