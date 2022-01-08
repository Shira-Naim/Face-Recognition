import cv2
import dlib
import face_recognition

#printing the versions
print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)

#loading the image to detect
image_test = cv2.imread('C:/Users/shira/CODE/images/testing/trump-modi.jpg')

#showing the current image with title
cv2.imshow("Image", image_test)

