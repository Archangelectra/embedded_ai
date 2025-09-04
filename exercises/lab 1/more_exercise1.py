import cv2
import numpy as np
import time

def adjust_brightness_contrast(img, alpha, beta):
   return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

image = cv2.imread("images/image.png")
cv2.imshow("Image (Before)", image)

image = adjust_brightness_contrast(image, 1.2, 30)

cv2.imshow("Image (After)", image)
cv2.waitKey()
