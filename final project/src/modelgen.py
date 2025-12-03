# imports
import numpy as np, tensorflow as tf
from tensorflow import keras

# basic test model i quickly threw together
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(112, 112, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(64, (2,2), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

# generic compilation
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

try:
    model.save("../model/model.keras") 
except:
    print("\nERROR: DIRECTORY NOT FOUND.\nThis program is coded to use the relative directory. Please rerun this program from within the /src folder.")