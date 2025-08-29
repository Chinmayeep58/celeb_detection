import numpy as np
import cv2
import seaborn as sn
import matplotlib.pyplot as plt

img=cv2.imread('./test_images/sg.jpg')
# print(img.shape)

# plt.imshow(img)
# plt.show()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray.shape)

plt.imshow(gray,cmap='gray')
# plt.show()

face_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade=cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

faces=face_cascade.detectMultiScale(gray,1.3,5)
print(faces)