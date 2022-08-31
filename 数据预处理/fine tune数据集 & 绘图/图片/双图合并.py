import numpy as np
import cv2


img1 = cv2.imread("pic-meld.png")
img2 = cv2.imread("pic-emorynlp.png")

imgs = np.hstack([img1, img2])
cv2.imwrite(".\\Concatenate words counts.jpg",imgs)