# 일반 ml keras
import tensorflow as tf
from tensorflow.keras import datasets

# 1. MNIST 데이터셋 임포트
mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
# 렐루 들어갈때 오버플로 안나게 하게끔 방지하력 255 나눠서 0~1 사이로 정규화
# 왜 255? 흑백은 할픽셀에 들어갈 수 있는 최대값이 255
# 흰색 0 검정새 255
# 그냥 데이터 좀 크다 싶으면  max 값으로 나눠버려

x_train, x_test = x_train/255.0, x_test/255.0

# 이미지 데이터
# x_train.shape
# (60000, 28, 28) => 갯수/y,x
# if 아이리스 데이터 (로우데이터)
# x_train.shape
# (150, 4) => 갯수/x


# 3. 모델 구성
# Flatten : 1차원으로 뿌림 28 by 28 은 1 by 7~로 바뀜. 인풋값은 명시 해줘야 됨.
# Sequential : 순서적으로. 순차적으로. ->,함수 내부에 층을 list로 넣어도 되고 add로 넣어도 되고
# add로 넣는게 나. 층을 변수로 할당하고 그 변수를 이용하는 경우도 생겨. 2번째 층이 5번째 가기도 하고 함.


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),#입력층에 활성화 함수를 안쓴거야
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 분류니까 마지막 은 소프트맥스
])
# => 층을 설정함(Sequential)
# in Layer -> relu -> hidden Layer -> softmax -> output Layer

# 히든층을 늘리면?
# 선 하나 더 늘어나서 좀더 비선형적으로 분류 가능하게 함
# 층 늘리면 단점 > vanishing gradient 기울기 소실
# 역전파시 기울기 소실 된다 업데이트가 잘 안된다.. 이거 정리 해놓자
# 렐루 - 시그모이드 선형/비선형 을 번갈아서 쓰면 좀 괜찮아지긴 함


# 모델에 대한 요약
# 층을 어떻게 나누었는지
# model.summary()

# 4. 모델 컴파일
model.compile(optimizer='adam', # 최적화 adam 아니면 adagrada
              loss='sparse_categorical_crossentropy', # softmax 랑 쓰면 고정 cross entropy
              metrics=['accuracy'])

# error 가 나오는 이유 > error가 갑자기 튄다. 그럼 이상한거야. call back 함수 써서 중간에 멈출 수 도 있어
# 아니면 error 튀기 전까지 epho 돌릴 수도 있어



# 5. 모델 훈련
model.fit(x_train, y_train, epochs=5)

# 6. 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test) #전체적인 결과
# predict 는 단건
print('테스트 정확도:', test_acc)