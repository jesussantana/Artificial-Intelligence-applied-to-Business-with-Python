# Image classification with Tensorflow and Keras
# ==============================================================================

import numpy
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, decode_predictions

# Create Instance
# ==============================================================================

iv3 = InceptionV3()

# Upload image
# ==============================================================================
from google.colab import files 
uploaded = files.upload()

x = image.img_to_array(image.load_img("car.jpg", target_size=(299,299)))

# Create Dimensions
# ==============================================================================

x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])

# Analyze the image with predict
# ==============================================================================

keras.applications.inception_v3.preprocess_input(x)
y = iv3.predict(x)

# Show result
# ==============================================================================

print(decode_predictions(y))

# Save prediction
# ==============================================================================

data1 = decode_predictions(y)
print("Image 2 classified as: ")
print(data1[0][0])