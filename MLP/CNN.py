import os
import shutil

import tensorflow as tf
import numpy as np
import cv2

wavData = "_train/wavData"
distDir = "_train/trainData"
modelDir = "clap_model"
try:
   shutil.rmtree(modelDir)
except:
    pass

# data Load
dList = os.listdir(distDir)
dList.sort(key=lambda x: int(x.split('.')[0]))
cntData = len(dList)
imgW = 0
imgH = 0
imgC = 0
for i in range(len(os.listdir(distDir))):
    img = cv2.imread(distDir + "/" + dList[i])
    imgH = img.shape[0]
    imgW = img.shape[1]
    imgC = img.shape[2]

train_x = np.zeros((cntData,) + (imgH,) + (imgW,) + (imgC,), dtype=np.uint8)
train_y = np.zeros(cntData, dtype=np.int32)

for i in range(len(dList)):
    train_x[i] = cv2.imread(distDir + "/" + dList[i])
    train_y[i] = dList[i].split('.')[0].split('_')[1]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imgH, imgW, imgC)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=20)

test_loss, test_acc = model.evaluate(train_x, train_y, verbose=2)
print(test_acc)
model.save(modelDir)