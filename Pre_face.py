from __future__ import absolute_import, division, print_function

# Thêm thư viện TensorFlow và tf.keras
import tensorflow
from keras.models import model_from_json
import numpy as np
import cv2
import collections
from tkinter import messagebox
from tkinter import *

count = 0
label = []
temp = 0

#import classifier for face and eye detection
face_class = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# Load json và tạo model
json_file = open('face_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

#Load trọng số đã train
model.load_weights('face_model_weights.h5')

#Tạo nhãn tương ứng với các số id
class_names = ['Tuyen', 'Thanh', 'Tan', 'Nga', 'Khanh', 'Chanh']

# Webcam setup for Face Detection
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # set Width
cap.set(4, 720) # set Height
while True:
    ret, frame = cap.read ()
    frame = cv2.flip(frame, 1) # Lật ảnh
    faces = face_class.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        y = y - 40
        h = h + 80
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        test_image = frame[y:y + h, x:x + w]

        #test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)     #Chuyển sang ảnh xám
        test_image = test_image/255.0      #Chuẩn hóa 0-255 thành 0-1
        test_image = cv2.resize(test_image, (130, 250))       #Chuyển về kích thước 28x28
        x_test = test_image.reshape(-1, 130, 250, 1)

        #Dự đoánmm
        predictions = model.predict(x_test)
        name = class_names[np.argmax(predictions[0])]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL = 4
        cv2.putText(frame, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        count += 1
        label.append(np.argmax(predictions[0]))
        print(np.amax(predictions[0]) * 100)

    cv2.imshow('Face', frame)

    # Tạo phím thoát
    k = cv2.waitKey(15) & 0xff
    if k == 27:  # Thoát khi nhấn phím ESC
        break
    elif count >= 9:  # Take 100 face sample and stop video
        if temp == 0:
            result = collections.Counter(label)
            id = result.most_common(1)[0][0]

            root = Tk()
            root.withdraw()
            messagebox.showinfo("Xác nhận !",'Chào bạn '+class_names[id]+', đã quét xong !')
            root.destroy()

            temp = 1
        else:
            if k == ord('e'):
                temp = 0
                count = 0
                label = []