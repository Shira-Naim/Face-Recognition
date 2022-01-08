# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 13:30:20 2021

@author: shira
"""
import face_recognition
import cv2

#loading image to detect
image_to_detect = cv2.imread('C:/Users/shira/CODE/images/testing/trump-modi.jpg')

#show the current image to detect
#cv2.imshow("test", image_to_detect)

#detect all faces in the image(hog\cnn models)
all_face_location = face_recognition.face_locations(image_to_detect, model='hog')

#print the number of face detected
print('There are {} faces in this image'.format(len(all_face_location)))

#looping through the face location
for index,current_face_location in enumerate(all_face_location):
    #splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,
                                                                      bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    cv2.imshow("Face no "+str(index+1),current_face_image)
    
    
