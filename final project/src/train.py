import numpy as np, tensorflow as tf
from tensorflow import keras
from dataloader import DigiFaceLoader

# load model
model = keras.models.load_model("../model/model.keras")

# instantiate the loader
loader = DigiFaceLoader(data_dir="../data")

# pull splits
train_ds = loader.get_dataset(validation_split=0.2, subset='training', seed=42)
val_ds = loader.get_dataset(validation_split=0.2, subset='validation', seed=42)

model.fit(train_ds, validation_data=val_ds,epochs=10)

model.save("..model/model.keras")