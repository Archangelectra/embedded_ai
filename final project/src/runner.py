import numpy as np, tensorflow as tf, opencv
import infer
from tensorflow import keras

model = keras.models.load_model("../model/model.keras")
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error occurred while reading frames, exiting.")
        break

    faces = infer(model)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break