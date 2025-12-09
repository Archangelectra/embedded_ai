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

# 2. Init Loader
loader = WiderFaceLoader(img_height=IMG_SIZE, img_width=IMG_SIZE, batch_size=BATCH_SIZE)
print("Loading WIDER FACE dataset...")
train_ds = loader.get_dataset(split='train')
val_ds = loader.get_dataset(split='validation')

# 3. Load Model
if os.path.exists("../model/mobilenet_face_detector.keras"):
    model = keras.models.load_model("../model/mobilenet_face_detector.keras")
else:
    from modelgen import create_transfer_model
    model = create_transfer_model((IMG_SIZE, IMG_SIZE, 3))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ---------------------------------------------------------
# PHASE 1: Train the Head (Feature Extraction)
# ---------------------------------------------------------
print("\n--- PHASE 1: Training Regression Head (Base Frozen) ---")

# We use append=False to overwrite any old file from previous runs
csv_logger_head = CSVLogger(results_file, append=False)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[csv_logger_head] # Add logger here
)

# ---------------------------------------------------------
# PHASE 2: Fine-Tuning (Train the whole model)
# ---------------------------------------------------------
print("\n--- PHASE 2: Fine-Tuning (Base Unfrozen) ---")
model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='mse',
    metrics=['mae']
)

# We use append=True to add these rows to the SAME file we just created
csv_logger_fine = CSVLogger(results_file, append=True)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FULL,
    callbacks=[csv_logger_fine] # Add logger here
)

# Save Final Model
model.save("../model/mobilenet_finetuned.keras")
print(f"Training Complete. Results saved to {results_file}")