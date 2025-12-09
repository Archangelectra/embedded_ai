import tensorflow as tf
from tensorflow import keras

def create_transfer_model(input_shape=(112, 112, 3)):
    # 1. Load the Base Model (MobileNetV2)
    # include_top=False removes the final classification layers (1000 classes)
    # weights='imagenet' loads the pre-trained weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    
    # 2. Freeze the Base Model
    # We don't want to destroy the pre-learned features during the first pass
    base_model.trainable = False
    
    # 3. Add Custom "Detection" Head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # Flatten the output
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # Output: 4 numbers (x, y, w, h) normalized between 0 and 1
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])
    
    return model

if __name__ == "__main__":
    model = create_transfer_model()
    
    # Compile with Regression loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    
    model.summary()
    model.save("../model/model.keras")
    print("Model created and saved to ../model/model.keras")