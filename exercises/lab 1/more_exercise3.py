import cv2
import numpy as np

image = cv2.imread("images/image.png")

for lo, hi in [(50,150), (100,200), (150,300)]:
    cv2.imshow(f"Canny {lo}-{hi}", cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), lo, hi))

cv2.waitKey()

# 50-150 did a terrible job, basically just scribbling all over the background leaves instead of the close up tree.
# 100-200 did an alright job, capturing more of the closer branches, still drawing the excess background leaves though.abs
# 150-300 did the best job, providing the most detail of the tree that's the primary subject, as well as even being able to properly outline some of the other background tree's branches, whereas 100-200 failed to properly do so