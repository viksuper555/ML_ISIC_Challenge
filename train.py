import os

from PIL import Image
from keras.applications.resnet_v2 import decode_predictions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_curve
from keras.preprocessing.image import ImageDataGenerator

train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 32

METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

# NasNet
# model = keras.Sequential([
#    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
#                   trainable=True),
#    layers.Dense(1, activation="sigmoid"),
# ])

model = keras.models.load_model("saved_model/")
# model.load_weights("saved_weights/")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=tf.float32,
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
    "data/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

validation_gen = validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)

model.fit(
    train_gen,
    epochs=20,
    # verbose=2,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    callbacks=[keras.callbacks.ModelCheckpoint("saved_model")],
)


def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.show()


test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

plot_roc(test_labels, test_gen)
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)
model.save("saved_model/")
