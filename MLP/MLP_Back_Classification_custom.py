import numpy as np

zero = np.array([    1, 1, 1
                    ,1, 0, 1
                    ,1, 0, 1
                    ,1, 0, 1
                    ,1, 1, 1], dtype="uint8")
one = np.array([[    0, 1, 0
                   ,1, 1, 0
                   ,0, 1, 0
                   ,0, 1, 0
                   ,1, 1, 1]], dtype="uint8")
two = np.array([[    1, 1, 1
                   ,0, 0, 1
                   ,1, 1, 1
                   ,1, 0, 0
                   ,1, 1, 1]], dtype="uint8")

train_x = np.array( [[1, 1, 0,1, 0, 1,1, 0, 1,1, 0, 1,1, 1, 1]# 0
                    ,[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]# 0
                    ,[0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]# 1
                    ,[1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]# 2
                    ,[0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]# 2
                    ,[1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]# 2
                    ,[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]# 2
                    ], dtype="uint8")
train_y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype="uint8")

test_x = np.array( [[1, 1, 1,
                     1, 0, 1,
                     1, 0, 1,
                     1, 0, 1,
                     1, 1, 1]# 0
                    ,[0, 1, 0,
                      1, 1, 0,
                      0, 1, 0,
                      0, 1, 0,
                      1, 1, 1]# 1
                    ,[1, 1, 1,
                      0, 0, 1,
                      1, 0, 1,
                      1, 0, 0,
                      1, 1, 1]# 2
                    ], dtype="uint8")
test_y = np.array([0, 1, 2], dtype="uint8")

train_y2 = np.zeros((train_y.shape[0],3), dtype=int)
for i in range(len(train_y)):
    train_y2[i][train_y[i]] = 1
train_y = train_y2

test_y2 = np.zeros((test_y.shape[0],3), dtype=int)
for i in range(len(test_y)):
    test_y2[i][test_y[i]] = 1
test_y = test_y2

# 층 별 노드 갯수 셋팅
iLnum = 15 # 입력층의 갯수
hLnum = 6 # 은닉층의 갯수
oLnum = 3 # 출력층의 갯수

iW = np.random.randn(hLnum,iLnum)
iB = np.random.randn(hLnum)

hW = np.random.randn(oLnum,hLnum)
hB = np.random.randn(oLnum)

epoch = 10
learnRate = 0.5
Error = np.zeros((epoch, train_x.shape[0]), dtype="float")

def w_sum(x, w, b):
    return np.sum(x * w, axis=1) + b

def relu(x):
    return np.maximum(0,x)

def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    if a.ndim == 1:
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    else:
        sum_exp_a = np.sum(exp_a, 1)
        sum_exp_a = sum_exp_a.reshape(sum_exp_a.shape[0], 1)
        y = exp_a / sum_exp_a
    return y
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def crossEntropyError(y, t): #y: 신경망 출력값 t: 정답
    delta = 1e-7 #아주 작은 값 (y가 0인 경우 -inf 값을 예방)
    return -np.sum(t*np.log(y+delta))

def Derivative_softmax_corssentropy(y,t):
    return y-t

def Derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

for ind_epoch in range(epoch):
    print(ind_epoch + 1, "번째 학습입니다.", ind_epoch + 1, "/", epoch)
    for index_train in range(len(train_x)):
        # FeedForword(순전파)
        # 은닉층 구현
        hSum = w_sum(train_x[index_train], iW, iB)
        hLayer = sigmoid(w_sum(train_x[index_train], iW, iB))

        # 출력층 구현
        oSum = w_sum(hLayer, hW, hB)
        oLayer = softmax(w_sum(hLayer, hW, hB))

        # 실제 값과 오차 구하기
        error = crossEntropyError(oLayer, train_y[index_train])
        Error[ind_epoch][index_train] = error

        # Back-Propagation(역전파)
        # 은닉층의 가중치와 편향 갱신
        # 손실함수와 비용함수 미분 : Derivative_softmax_corssentropy
        temphW = hW  # 다음층 갱신을 위해 저장
        a2 = Derivative_softmax_corssentropy(oLayer, train_y[index_train])
        hB = hB - learnRate * a2
        hW = hW - (learnRate * a2.reshape(a2.shape[0], 1) * hLayer)

        # 입력층의 가중치와 편향 갱신
        # 은닉층을 갱신했던 미분값
        # 갱신전 은닉층의 가중치
        a1 = np.sum(a2.reshape(a2.shape[0], 1) * temphW * Derivative_sigmoid(hSum), axis=0)
        iB = iB - learnRate * a1
        iW = iW - learnRate * a1.reshape(a1.shape[0], 1) * train_x[index_train]

    print(ind_epoch + 1, "번째 학습 오류 :: ", np.sum(Error[ind_epoch]) / Error.shape[1])

#테스트 시작
testerr = 0
for index_test in range(len(test_x)):
    # FeedForword(순전파)
    # 은닉층 구현
    hLayer = sigmoid(w_sum(test_x[index_test], iW, iB))

    # 출력층 구현
    oLayer = softmax(w_sum(hLayer, hW, hB))
    print(oLayer)
    if (str(np.argmax(oLayer)) != str(np.argmax(test_y[index_test]))):
        testerr += 1

print("테스트 오류율 ::", testerr / len(test_x) * 100)