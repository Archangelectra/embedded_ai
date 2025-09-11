import cv2
import random
import numpy as np



def add_salt_and_pepper_noise(image, prob):

        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob / 2:  # Pepper noise (set to 0)
                    output[i][j] = 0
                elif rdn > thres + (prob / 2):  # Salt noise (set to 255)
                    output[i][j] = 255
                else:  # Keep original pixel value
                    output[i][j] = image[i][j]
        return output

kernel = np.ones((3,3), np.uint8)
image = cv2.imread("images/grayscaleimage.png")
cv2.imshow("Default Image", image)

base_image = image

image = add_salt_and_pepper_noise(image,0.25)
cv2.imshow("Noised Image", image)

image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow("Denoised Image", image)

cv2.waitKey()

print(f"PSNR: {cv2.PSNR(base_image, image)}")
