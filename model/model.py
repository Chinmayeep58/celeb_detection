import numpy as np
import cv2
import seaborn as sn
import matplotlib.pyplot as plt
import os

img=cv2.imread('./test_images/sg.jpg')
# print(img.shape)

# plt.imshow(img)
# plt.show()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray.shape)

# plt.imshow(gray,cmap='gray')
# plt.show()

face_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

faces=face_cascade.detectMultiScale(gray,1.3,5)
# print(faces)

(x,y,w,h)=faces[0]
face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# plt.imshow(face_img)
# plt.show()

cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=face_img[y:y+h,x:x+w]
    eyes=eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
plt.figure()
# plt.imshow(face_img,cmap='gray')
plt.imshow(roi_color,cmap='gray')
# plt.show()


def get_cropped_img_if_2_eyes(img_path):
    img=cv2.imread(img_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            return roi_color
        
original=cv2.imread('./test_images/sg.jpg')
plt.imshow(original)

cropped=get_cropped_img_if_2_eyes('./test_images/sg.jpg')
plt.imshow(cropped)
# plt.show()

path_to_data="./dataset/"
path_to_cr_data="./dataset/cropped/"

img_dirs=[]
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

# print(img_dirs)
