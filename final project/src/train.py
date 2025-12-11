import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from dataloader import WiderFaceLoader
import os

# CONFIG
IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FULL = 10

# 1. setup logging
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, "results.csv")

# 2. instantiate loaders (train & val)
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

# 3. create new model
print("Creating new model from modelgen...")
from modelgen import create_transfer_model
model = create_transfer_model((IMG_SIZE, IMG_SIZE, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss='mse', 
    metrics=['mae']
)

# ---------------------------------------------------------
# PHASE 1: train head
# ---------------------------------------------------------
print("\n--- PHASE 1: Training Regression Head ---")
csv_logger = CSVLogger(results_file, append=False)

# save the best model automatically
checkpoint = ModelCheckpoint(
    "../model/best_head.keras", 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS_HEAD, 
    callbacks=[csv_logger, checkpoint]
)

# ---------------------------------------------------------
# PHASE 2: fine tuning
# ---------------------------------------------------------
print("\n--- PHASE 2: Fine-Tuning ---")
# unfreeze the base model
model.trainable = True

# recompile with very low learning rate for fine tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='mse', 
    metrics=['mae']
)

csv_logger = CSVLogger(results_file, append=True)
checkpoint_tuned = ModelCheckpoint(
    "../model/modeltuned.keras", 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS_FULL, 
    callbacks=[csv_logger, checkpoint_tuned]
)

print("Training Complete. Use '../model/modeltuned.keras' for inference.")