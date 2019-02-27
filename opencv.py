# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:54:39 2019

@author: akash
"""

import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("./saved-model/fer-124epoch.h5")

face_cascade = cv2.CascadeClassifier("frontalface_default.xml")
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_new = cv2.resize(gray, (48, 48))
    img_new = img_new.reshape(-1, 48, 48, 1)
    
    score = model.predict(img_new)
    emotiontext = (emotions[np.argmax(score)])

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 4)
        cv2.putText(img, emotiontext, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    cv2.imshow('img', img)    
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()