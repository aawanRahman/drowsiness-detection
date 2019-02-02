# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:23:33 2018

@author: ASUS
"""

import cv2
import numpy as np
import dlib  
import imutils
import time
import winsound 

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = load_model('cnn_model_eye.h5')
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
cascPath = 'haarcascade_frontalface_default.xml'  
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

LEFT_EYEBROW_POINTS = list(range(17, 22)) 
LEFT_EYE_POINTS = list(range(36, 42))

faceCascade = cv2.CascadeClassifier(cascPath)  
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)
ans = 0
face_detect = 0 
blink_rate = 0
blink = 0
flag = True
start_time = 0
end_time = 0
total_time = 0 
 
while(cap.isOpened()):
    global ans
    global blink_rate
    global blink
    global flag
    global face_detect
    a = time.time()
    ret, image = cap.read()
 
    faces = faceCascade.detectMultiScale(  
      image,  
      scaleFactor=1.05,  
      minNeighbors=5,  
      minSize=(100, 100),  
      flags=cv2.CASCADE_SCALE_IMAGE  
    )  
       

    print("Founded {0} faces!".format(len(faces)) )  
    
 #if face not found..
    if(len(faces) ==0 ) :
        face_detect = face_detect + 1
        if (face_detect > 20 ) :
            print("unawareness detection or drowsiness detected.")
            frequency =2500
            duration = 1000
            winsound.Beep(frequency , duration )
            face_detect=0
    else :
        
# if face detected the then further processing   
        for (x, y, w, h) in faces:  
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
      
            landmarks = np.matrix([[p.x, p.y]
            for p in predictor(image, dlib_rect).parts()])  
      
            landmarks_display = landmarks[LEFT_EYE_POINTS + LEFT_EYEBROW_POINTS]  
            for idx, point in enumerate(landmarks_display):
                pos = (point[0, 0], point[0, 1])
            (x, y, w, h) = cv2.boundingRect(landmarks_display)
            roi1 = image[y:y + h, x:x + w]
            roi1 = imutils.resize(roi1, width=250, height=250, inter=cv2.INTER_CUBIC)

        cv2.imwrite('temp.jpg', roi1)
        img = cv2.imread('temp.jpg')
        img = cv2.resize(img,(64,64))
        img = np.reshape(img,[1,64,64,3])
        classes1 = model.predict_classes(img)
        time.sleep(1)
        print(classes1)
        print(" blink --> " , blink)
        
        if classes1 == 0 :
            ans=ans+1
            if flag == True :
                blink = blink + 1
                flag = False
                total_time = total_time + (end_time - start_time) 
                start_time= time.time()
                if blink >=17 :
                    if total_time < 60.0 :
                        frequency =2500
                        duration = 1000
                        winsound.Beep(frequency , duration )
                        print("drowsiness detected")
                        print(blink)
                        print("total time" , total_time)
                        total_time = 0
                        blink = 0
                         
                    elif total_time > 60 :
                         blink = 0
                         total_time = 0
                
        elif classes1== 1:
            ans =0
            flag = True
            end_time = time.time()
            
        if ans >= 5 :
            frequency =2500
            duration = 1000
            winsound.Beep(frequency , duration )
            print("drowsiness detected")
       
            ans=0

            
        b = time.time()
        print(b-a)
        time.sleep(.20)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
