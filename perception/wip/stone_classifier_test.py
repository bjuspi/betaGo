import numpy as np
import tensorflow as tf
from keras.preprocessing import image

reconstructed_model = tf.keras.models.load_model("cnn.h5")

test_dir = "image/sample/training/white/9fccf5eb-1f2e-11ec-be46-4074e0e771b4.jpg"

img = image.load_img(test_dir, target_size=(30, 30))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = reconstructed_model.predict(images)
print(classes)