import cv2
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# Path for face image database
path = 'dataset'
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        img_numpy = img_numpy/ 255.0
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(cv2.resize(img_numpy, (130, 250)))
        ids.append(id)
    return faceSamples, ids

faces, ids = getImagesAndLabels(path)
faces, ids = shuffle(faces, ids)
x_train, x_test, y_train, y_test = train_test_split(faces, ids, test_size=0.33, random_state=0)

x_train = np.reshape(x_train, [-1, 130, 250, 1])
y_train = to_categorical(y_train)
x_test = np.reshape(x_test, [-1, 130, 250, 1])
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
for j in range(5):
    # create model
    model = Sequential()

    # add model layers
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(130, 250, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=10)

    # Lưu model
    model_json = model.to_json()
    with open("face_model.json", "w") as json_file:
        json_file.write(model_json)

    # Lưu các trọng số
    model.save_weights('face_model_weights_'+str(j)+'.h5')



