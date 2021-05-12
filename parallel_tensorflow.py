import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras import layers

# tf.device(`GPU:0`): ...
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = EfficientNetB0(input_shape=(64,64,3))
    model.compile()

