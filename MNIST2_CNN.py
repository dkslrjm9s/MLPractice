import tensorflow as tf
import numpy as np


# mnist2 = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = mnist2.load_data()
from dkdk import mnist2Data
x_train, y_train, x_test, y_test = mnist2Data()

x_train, x_test = x_train/255.0, x_test/255.0


# 입력 모양 설정
sample_shape = x_train[0].shape
img_width, img_height = sample_shape[1], sample_shape[0]
input_shape = (img_height, img_width, 1)

# 데이터 재구성
x_train = x_train.reshape(len(x_train), input_shape[0], input_shape[1], input_shape[2])
x_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten()) # con pool 하고 나온 1차원 데이터를 flatten 에 밀어넣음 1차원까지 안나와도 그냥 flatten 하면 1차원으로 다 바꿈
model.add(tf.keras.layers.Dense(576, activation='relu'))
model.add(tf.keras.layers.Dense(288, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)

print(1)