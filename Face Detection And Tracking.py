#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
alg = "haarcascade_frontalface_default.xml"
haarcsc = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read(0)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haarcsc.detectMultiScale(grayimg,1.3,5)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),4)
        
    cv2.imshow("FaceDetect",img)
    
    key = cv2.waitKey(10)
    print(key)
    
    if key == ord('a'):
        break
        
cam.release()
cv2.destroyAllWindows()

