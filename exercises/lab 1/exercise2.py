# Completes exercise 2 in the lab

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("images/image.png")

image = np.array(image)
image = image[:, :, [2, 1, 0]] 

plt.imshow(image)
plt.show()
