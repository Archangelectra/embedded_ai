import cv2
import time

image = cv2.imread("images/image.png")
image = cv2.putText(image, "Lorem Ipsum", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

cv2.imshow("Image", image)

print("Press Q to quit, S to save.")

while(True): 
    key = cv2.waitKey()
    print(key)
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        print("Saving at new_image.png")
        cv2.imwrite("new_image.png", image)
        break
    elif key == 81:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("Image", image)
    elif key == 83:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Image", image)

cv2.destroyAllWindows()

