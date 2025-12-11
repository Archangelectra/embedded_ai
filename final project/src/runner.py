import cv2
import tensorflow as tf
from tensorflow import keras
from infer import infer
import sys
import numpy as np

# -----------------------------------------------------------------------------
# HELPER CLASS: box smoother
# -----------------------------------------------------------------------------
class BoxSmoother:
    def __init__(self, alpha=0.7):
        """
        Smoothing factor alpha:
        - 0.1: Very smooth, slow reaction (high latency).
        - 0.9: Very jittery, fast reaction (low latency).
        - 0.2 is usually a good sweet spot.
        """
        self.alpha = alpha
        self.last_box = None

    def update(self, new_box):
        x, y, w, h = new_box
        
        # if this is the first frame, just accept the box as is
        if self.last_box is None:
            self.last_box = [float(x), float(y), float(w), float(h)]
            return new_box

        # smooth each coordinate independently using exponential moving average
        self.last_box[0] = self.last_box[0] * (1 - self.alpha) + x * self.alpha
        self.last_box[1] = self.last_box[1] * (1 - self.alpha) + y * self.alpha
        self.last_box[2] = self.last_box[2] * (1 - self.alpha) + w * self.alpha
        self.last_box[3] = self.last_box[3] * (1 - self.alpha) + h * self.alpha

        # return as integers for drawing
        return (int(self.last_box[0]), int(self.last_box[1]), 
                int(self.last_box[2]), int(self.last_box[3]))

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# 1. load model
print("Loading model...")
try:
    # try loading the fine-tuned model first
    model = keras.models.load_model("../model/modeltuned.keras")
except Exception:
    try:
        # fallback to the head-only model
        print("Fine-tuned model not found, trying 'best_head.keras'...")
        model = keras.models.load_model("../model/best_head.keras")
    except Exception:
        # fallback to the original filename (from earlier versions)
        print("Trying original filename 'model.keras'...")
        try:
            model = keras.models.load_model("../model/model.keras")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

print("Model loaded successfully.")

# 2. initialize USB webcam
CAMERA_INDEX = 0
print(f"Initializing USB Camera at index {CAMERA_INDEX}...")

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print(f"Error: Could not open video device {CAMERA_INDEX}.")
    print("Try running 'ls -l /dev/video*' to see available devices.")
    sys.exit(1)

# PERFORMANCE TIP: Force MJPG format for USB cameras to get higher FPS
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. setup video writer (backup)
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc('M','J','P','G'), 
    10, 
    (frame_width, frame_height)
)

# 4. initialize the smoother
smoother = BoxSmoother(alpha=0.25)

print("\n--- RUNNING WITH DISPLAY ---")
print("Reading from USB Camera.")
print("Press 'q' to quit or Ctrl+C to stop.\n")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            break

        # inference
        # returns a list of tuples: [(x, y, w, h)]
        faces = infer(model, frame)

        # process detections
        if len(faces) > 0:
            raw_box = faces[0]
            
            # --- APPLY SMOOTHING ---
            (x, y, w, h) = smoother.update(raw_box)
            
            # 1. draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 2. print label
            label = f"Face: {w}x{h}"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        else:
            # if no face found, reset the smoother so it doesn't "fly" in from the old position next time
            smoother.last_box = None

        # save frame to video file (Backup)
        out.write(frame)
        
        # --- GUI DISPLAY ---
        cv2.imshow('Face Detection', frame)

        # check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    print("Cleaned up. Video saved to 'output.avi'.")