import cv2
import numpy as np

image = cv2.imread("images/image.png")

small_nn  = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
small_lin = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
small_cub = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
large_nn  = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
large_lin = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
large_cub = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Nearest", small_nn); cv2.imshow("Linear", small_lin); cv2.imshow("Cubic", small_cub)

cv2.waitKey()

cv2.destroyAllWindows()

cv2.imshow("Nearest", large_nn); cv2.imshow("Linear", large_lin); cv2.imshow("Cubic", large_cub)

cv2.waitKey()

# Linear has a lot more blurriness, caused by what appears to be anti aliasing.
# Cubic appears to cause a ton of pixelation when it blows up the image.abs
# Nearest seems to be fine, with very little artifacting from what I can see. It's a bit blurry, but I think that's due to the image's base resolution.