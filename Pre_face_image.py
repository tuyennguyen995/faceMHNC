from __future__ import absolute_import, division, print_function

# Thêm thư viện TensorFlow và tf.keras
import tensorflow
from keras.models import model_from_json
import numpy as np
import cv2
import matplotlib.pyplot as plt
#Thư viện open file
from tkinter import filedialog
from tkinter import *

#import classifier for face and eye detection
face_class = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# Load json và tạo model
json_file = open('face_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

#Load trọng số đã train
model.load_weights('face_model_weights.h5')

#Tạo nhãn tương ứng với các số từ 0-9
class_names = ['Tuyen', 'Tha2nh', 'Tan', 'Nga', 'Khanh', 'Chanh']
#Chọn file ảnh
root = Tk()
root.withdraw()
root.filename = filedialog.askopenfilename(initialdir = "/",title = "Chọn file ảnh",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

#Xử lý ảnh đầu vào
img = cv2.imread(root.filename)    #load ảnh
faces = face_class.detectMultiScale(img, 1.3, 5)
for (x, y, w, h) in faces:
    y = y - 40
    h = h + 80
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    test_image = img[y:y + h, x:x + w]

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)     #Chuyển sang ảnh xám
    test_image = test_image/255.0      #Chuẩn hóa 0-255 thành 0-1
    test_image = cv2.resize(test_image, (130, 250))       #Chuyển về kích thước 28x28
    x_test = test_image.reshape(-1, 130, 250, 1)

    #Dự đoánmm
    test_image = (np.expand_dims(x_test, 0))
    predictions = model.predict(x_test)
print(class_names[np.argmax(predictions[0])])
plt.imshow(img)
plt.show()
root.destroy()