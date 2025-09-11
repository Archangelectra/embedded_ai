import cv2
import numpy as np
import time

def adjust_brightness_contrast(img_rgb: np.ndarray, alpha: float=1.0, beta: float=0.0):
    out = np.clip(alpha*img_rgb + beta, 0, 255).astype(np.uint8)
    return out

image = cv2.imread("images/image.png")
cv2.imshow("Image (Before)", image)

image = adjust_brightness_contrast(image, 1.2, 30)

cv2.imshow("Image (After)", image)
cv2.waitKey()
