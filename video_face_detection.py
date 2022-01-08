# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:20:16 2021

@author: shira
"""
import face_recognition
import cv2

# 0 means the default camera
webcam_video_stream = cv2.VideoCapture('C:/Users/shira/CODE/images/testing/modi.mp4')

#intialize empty array for face locations
all_face_locations = []

while True:
    #get current frame
    ret,current_frame = webcam_video_stream.read()
    
    #resize the current frame to 1/4 size to proces paster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    
    #detect all faces in the image(hog\cnn models)
    all_face_location = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2, model='hog')
    
    #looping through the face locations
    for index,current_face_location in enumerate(all_face_location):
        #splitting the tuple to get the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4  
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,
                                                                      bottom_pos,left_pos))
        #draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos), (0,0,255),2)
    #showing the current face with rectangle drawn
    cv2.imshow("Webcam Video",current_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
#once loop break release the camera&close the windows
webcam_video_stream.release()
cv2.destroyAllWindows()
    