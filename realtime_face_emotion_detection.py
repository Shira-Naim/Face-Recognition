# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:20:16 2021

@author: shira
"""
import cv2
import time
import numpy as np
import face_recognition
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

#capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)


#face expression model intializition
face_exp_model = model_from_json(open("C:/Users/shira/CODE/dataset/facial_expression_model_structure.json","r").read())
#load weights into model
face_exp_model.load_weights("C:/Users/shira/CODE/dataset/facial_expression_model_weights.h5")
#list of emotion labels
emotions_label = ('angry','disgust','fear','happy','sad','surprise','neutral')

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
        
        #---Face detection
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
                
        #---Face Rectangle
        #draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos), (0,0,255),2)
        
        #---Emotion Detection
        #preprocess input,convert it to an image like as the data in the dataset
        #convert to grayscale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        #resize to 48x48 px size
        current_face_image = cv2.resize(current_face_image,(48,48))
        #convert the PIL image into a 3d numpy array
        img_pixels = image.img_to_array(current_face_image)
        #expend the shape of an array into single row multiple columns
        img_pixels = np.expand_dims(img_pixels, axis=0)
        #pixels are in range of [0,255]. normalize in scale of [0,1]
        img_pixels /= 255
        
        
        #---Prediction
        #do prediction using model, get the prediction values for all 7 expressions
        exp_predictions = face_exp_model.predict(img_pixels)
        #find max indexed prediction value(0 till 7)
        max_index = np.argmax(exp_predictions[0])
        #get correspondind label from emotion labels
        emotions_label = emotions_label[max_index]
        
        
        #---Results
        #display the name as text in the img
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,emotions_label,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
                 
    #display the video
    cv2.imshow("Webcam Video",current_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
#once loop break release the camera&close the windows
webcam_video_stream.release()
cv2.destroyAllWindows()
    