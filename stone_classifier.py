import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

training_dir = "image/sample/training"
training_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = training_datagen.flow_from_directory(
    training_dir,
    target_size=(30, 30),
    class_mode='categorical',
    batch_size=32
    )

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(30, 30, 3)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # # The second convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # # The third convolution
    # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # # The fourth convolution
    # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=5, steps_per_epoch=5, verbose = 1)

model.summary()

model.save("cnn.h5")