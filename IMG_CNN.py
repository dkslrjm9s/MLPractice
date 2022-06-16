

import cv2
import numpy as np
import os
import tensorflow as tf

fnList = os.listdir("scr_data")
x_train = np.ones((10, 20, 8, 3), dtype="uint8")
y_train = []

for fnind in range(len(fnList)):
    img = cv2.imread("scr_data/" + fnList[fnind])
    x_train[fnind] = np.append(img,
                        np.full((img.shape[0],8-img.shape[1], img.shape[2]), 248, dtype="uint8"), axis=1)
    y_train.append(fnList[fnind].replace(".png",""))

x_train, x_test = x_train /255.0, x_train /255.0
y_train, y_test = np.array(y_train, dtype="uint8"), np.array(y_train, dtype="uint8")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20,8,3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten()) # con pool 하고 나온 1차원 데이터를 flatten 에 밀어넣음 1차원까지 안나와도 그냥 flatten 하면 1차원으로 다 바꿈
model.add(tf.keras.layers.Dense(448, activation='relu'))
model.add(tf.keras.layers.Dense(224, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print(1)