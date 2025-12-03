import tensorflow as tf
from tensorflow import keras
from dataloader import WiderFaceLoader

# 1. load model
model = keras.models.load_model("../model/face_detector.keras")

# 2. init loader
loader = WiderFaceLoader(img_height=112, img_width=112)

print("Loading/Downloading WIDER FACE (this may take time on first run)...")
train_ds = loader.get_dataset(split='train')
val_ds = loader.get_dataset(split='validation')

# 3. compile, train.
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("../model/face_detector_trained.keras")