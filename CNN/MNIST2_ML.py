import tensorflow as tf
import numpy as np

mnist2 = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist2.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = y_train, y_test

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),#입력층에 활성화 함수를 안쓴거야
    tf.keras.layers.Dense(392, activation=tf.nn.relu),
    tf.keras.layers.Dense(196, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 분류니까 마지막 은 소프트맥스
])

model.compile(optimizer='adam', # 최적화 adam 아니면 adagrada
              loss='sparse_categorical_crossentropy', # softmax 랑 쓰면 고정 cross entropy
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('테스트 정확도:', test_acc)