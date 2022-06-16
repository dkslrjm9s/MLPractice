import tensorflow as tf
from tensorflow.keras import datasets

# 1. MNIST 데이터셋 임포트
mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential()
# Y,X, 채널
# input_shape=(28, 28, 3)

# CNN 시작

# Convolusion : 필터가 원본이미지에 붙음 필터에 원본이미지 픽셀의 곱에 합으로 convolved feature 을 만듬
# 원본이미지에서 필터와 비슷한 구간의 확휼이 convolved feature에 저장함
# 이런 특징 필터가 갯수가 엄청 많아
# convolusion 필터씌운 결과는 원본이미지보다 차원이 무조건 줌 (1 by 1 쓴거 아니면)
# 이런식으로 직선 검출, 곡선 검충 등 필터는 보통 30개 정도씀 그럼 하나의 이미지에서 30개 가 나오고
# 그 결과에서 곡선 검출 한거에서 직선이 있었나 ? 해서 결과에 또 필터를 씀
# 그런식으로 결과값들이 엄청 많이 나옴
# 필터에 대한 특징을 찾는거!! 곡 + 직 / 직 + 곡은 다른애야! 순서가 아얘달라!
# convolusion : 데이터 수를 늘려. 특징별로

# padding : 필터 결과값을 원본이미지 차원이랑 동일하게 하고 싶어. 그럼 원본이미지 가장자리를 늘려. 0/1로 채울지는 상황별로

# pooling
# max, avr, min 있는데 보통 max
# pooling 필터 내에 max, avr, min  값을 선정함. 차원은 무조건 줄어듬
# stride : 몇칸 건너뛸건가 최소 1. x,y 둘다 적어야 해.

# max 풀링 쓰는 이유?
# avr 는 평균이라서 좀 괜찮아 성능은 근데 시간이 너무 오래 걸려
# convolusion 들어가면 한 이미지에 15만 개 막 나오는데 언제 다 계산해
# max 풀링 왜 쓰냐면 계산 쉽고 어쨌던 특징은 무조건 살아. 강조

# con(갯수가 늘어남) -> pool(크기가 줌) -> con -> pool >> 1차원 데이터가 나올거라고 기대함
# 28 x 28 x 3(채널) x20(필터수) -> 14 x 14 x 3 x20 -> 14 x 14 x 3 x20 x30(필터수) ->
#https://techblog-history-younghunjo1.tistory.com/125

# 1SET
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# 2SET
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# 3SET
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# CNN 끝

model.add(tf.keras.layers.Flatten()) # con pool 하고 나온 1차원 데이터를 flatten 에 밀어넣음 1차원까지 안나와도 그냥 flatten 하면 1차원으로 다 바꿈
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
print(test_acc)