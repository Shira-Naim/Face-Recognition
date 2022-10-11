import cv2
import dlib
import keras
import face_recognition
import tensorflow

# printing the versions
print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)
print(keras.__version__)
print(tensorflow.__version__)

# loading the image to detect
image_test = cv2.imread('some-directory/Face-Recognition/images/testing/trump-modi.jpg')

# showing the current image with title
cv2.imshow("Image", image_test)

