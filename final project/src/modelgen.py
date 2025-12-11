import tensorflow as tf
from tensorflow import keras

def create_transfer_model(input_shape=(112, 112, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(4, activation='linear') 
    ])
    
    return model

if __name__ == "__main__":
    model = create_transfer_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    model.save("../model/model.keras")
    print("Model created and saved to ../model/model.keras")