import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from dataloader import WiderFaceLoader
import os

# CONFIG
IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FULL = 5

# 1. Setup Logging
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, "results.csv")

# 2. Instantiate Loaders (Train & Val)
print("Loading Training Data...")
train_loader = WiderFaceLoader(
    data_dir="../data/WIDER_train/images",
    annotation_file="../data/wider_face_split/wider_face_train_bbx_gt.txt",
    img_height=IMG_SIZE,
    img_width=IMG_SIZE,
    batch_size=BATCH_SIZE
)
train_ds = train_loader.get_dataset()

print("Loading Validation Data...")
val_loader = WiderFaceLoader(
    data_dir="../data/WIDER_val/images",
    annotation_file="../data/wider_face_split/wider_face_val_bbx_gt.txt",
    img_height=IMG_SIZE,
    img_width=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = val_loader.get_dataset()

# 3. Load or Create Model
if os.path.exists("../model/mobilenet_face_detector.keras"):
    print("Loading existing model...")
    model = keras.models.load_model("../model/mobilenet_face_detector.keras")
else:
    print("Creating new model from modelgen...")
    from modelgen import create_transfer_model
    model = create_transfer_model((IMG_SIZE, IMG_SIZE, 3))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ---------------------------------------------------------
# PHASE 1: Train Head
# ---------------------------------------------------------
print("\n--- PHASE 1: Training Regression Head ---")
csv_logger = CSVLogger(results_file, append=False)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=[csv_logger])

# ---------------------------------------------------------
# PHASE 2: Fine-Tuning
# ---------------------------------------------------------
print("\n--- PHASE 2: Fine-Tuning ---")
model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

csv_logger = CSVLogger(results_file, append=True)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FULL, callbacks=[csv_logger])

model.save("../model/mobilenet_finetuned.keras")
print("Training Complete.")