import os

import numpy as np
import tensorflow as tf
from PIL import Image


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

model = tf.keras.models.load_model("saved_model/")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=[tf.keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
base_dir = os.path.abspath(os.path.dirname(__file__))

def load(filename):
   np_image = Image.open(filename)
   np_image = np_image.resize((224, 224))
   np_image = np.array(np_image).astype('float32')/255
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


def predict_img(filename):
    target = os.path.join(base_dir,'temp/' + filename)
    image = load(target)
    prediction = model.predict(image)[0][0] * 100

    if prediction < 50:
        result = 'Ракът е доброкачествен. Коефициент на сигурност - ' + str((100-prediction))
    else:
        result = 'Ракът е злокачествен. Коефициент на сигурност - ' + str(prediction)

    return result