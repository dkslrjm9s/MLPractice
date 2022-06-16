import numpy as np

# 아이리스 데이터 로드 및 트레인 set과 test set 분리
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

#train_test_split(x,y, test_size,shuffle,stratify,random_state)
#
# test_size: 테스트 셋 구성의 비율을 나타냅니다. train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다.
# shuffle: default=True 입니다. split을 해주기 이전에 섞을건지 여부입니다. 보통은 default 값으로 놔둡니다.
# stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값입니다. stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. (한 쪽에 쏠려서 분배되는 것을 방지합니다) 만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.
# random_state: 세트를 섞을 때 해당 int 값을 보고 섞으며, 하이퍼 파라미터를 튜닝시 이 값을 고정해두고 튜닝해야 매번 데이터셋이 변경되는 것을 방지할 수 있습니다.
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.3, stratify = iris.target,random_state=42)

# 비율대로 training set과 validation set이 잘 나뉘어짐 확인
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
#(105, 4) (45, 4) (105,) (45,)

# 입력층과 출력층의 형태 확인
print(train_x[0].shape, train_x[0])
#(4,) [5.1 2.5 3.  1.1]
print(train_y[0].shape, train_y[0])
#() 1


# 입력층은 x의 배열 크기로 정하고
# 은닉층은 입력층과 출력층의 갯수를 고려하여 임의로 정하고
# 출력층은 아래와 같이 정하겠다.
# iris 데이터에서는 총 세가지의 target data가 나올 수있다.
# 0:setosa, 1:versicolor, 2:virginica
# 우리는 분류로 모델의 결과값을 출력할 것이기 때문에 총 세개의 출력층이 구성된다.
# 하지만 위와 같이 현재 iris.targer은 하나의 변수만 출력 됨으로 임의로 세개의 노드가 출력 될 수 있도록 바꿔준다.
#(LabelEncoder) 그 번호를 원핫인코딩 방식 >> 라이브러리 쓰면 이런식으로도 할 수 있데
train_y2 = np.zeros((train_y.shape[0],3), dtype=int)
for i in range(len(train_y)):
    train_y2[i][train_y[i]] = 1
train_y = train_y2

test_y2 = np.zeros((test_y.shape[0],3), dtype=int)
for i in range(len(test_y)):
    test_y2[i][test_y[i]] = 1
test_y = test_y2

# 결과 :
# 이전 : 1        | 2
# 이후 : [0,1,0]  | [0,0,1]

# 층 별 노드 갯수 셋팅
iLnum = 4 # 입력층의 갯수
hLnum = 2 # 은닉층의 갯수
oLnum = 3 # 출력층의 갯수
# 층 별 기본 설정 셋팅
#입력층
# input - hidden layer
# w : 다음층 갯수 by 현재 층 갯수 행렬 (h by w / y by x)
# b : 다음층 갯수
# h(actuvation) : 렐루 함수
iW = np.random.randn(hLnum,iLnum)
# iW = np.array([[ 0, 0, 0, 0],
#        [ 1, 1, 1, 1],
#        [ 2, 2, 2, 2]])
# array([[ 1.16384755, -0.63122363, -0.35677184, -1.07867945],
#        [ 0.47896488, -1.90171631,  1.08599673,  1.23527132],
#        [-0.45043781, -1.35136725,  1.04224091,  1.1780769 ]])
iB = np.random.randn(hLnum)
# iB = np.array([ 1,  2, 3 ])
# [ 0.52735944  1.45736779 -0.9192674 ]
#은닉층
# hidden - output layer
# w : 다음층 갯수 by 현재 층 갯수 행렬 (h by w / y by x)
# b : 다음층 갯수
# h(actuvation) : 소프트맥스 함수
hW = np.random.randn(oLnum,hLnum)
hB = np.random.randn(oLnum)

#출력층 3개
# 손실함수 : 엔트로피

epoch = 100
learnRate = 0.7
Error = np.zeros((epoch, train_x.shape[0]), dtype="float")

#axis가 1 이면 행끼리 더하는 것이고, axis가 0 이면 열끼리 더하는 것이다.
def w_sum(x, w, b):
    return np.sum(x * w, axis=1) + b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def crossEntropyError(y, t): #y: 신경망 출력값 t: 정답
    delta = 1e-7 #아주 작은 값 (y가 0인 경우 -inf 값을 예방)
    return -np.sum(t*np.log(y+delta))

def Derivative_softmax_corssentropy(y,t):
    return y-t

def Derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def Derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

for ind_epoch in range(epoch):
    for index_train in range(len(train_x)):
        # FeedForword(순전파)
        #은닉층 구현
        hSum = w_sum(train_x[index_train], iW, iB)
        hLayer = sigmoid(w_sum(train_x[index_train], iW,iB))

        #출력층 구현
        oSum = w_sum(hLayer, hW, hB)
        oLayer = softmax(w_sum(hLayer, hW, hB))

        #실제 값과 오차 구하기
        error = crossEntropyError(oLayer, train_y[index_train])
        Error[ind_epoch][index_train] = error

        # Back-Propagation(역전파)
        # 은닉층의 가중치와 편향 갱신
        # 손실함수와 비용함수 미분 : Derivative_softmax_corssentropy
        temphW = hW #다음층 갱신을 위해 저장
        a2 = Derivative_softmax_corssentropy(oLayer,train_y[index_train])
        hB = hB - learnRate * a2
        hW = hW - (learnRate * a2.reshape(a2.shape[0], 1) * hLayer)

        # 입력층의 가중치와 편향 갱신
        # 은닉층을 갱신했던 미분값
        # 갱신전 은닉층의 가중치
        a1 = np.sum(a2.reshape(a2.shape[0], 1) * temphW * Derivative_sigmoid(hSum), axis=0)
        iB = iB - learnRate * a1
        iW = iW - learnRate * a1.reshape(a1.shape[0], 1) * train_x[index_train]


    print(ind_epoch + 1, "번째 학습입니다.", ind_epoch+1 , "/", epoch)
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
    print(str(np.argmax(test_y[index_test])))
    if (str(np.argmax(oLayer)) != str(np.argmax(test_y[index_test]))):
        testerr += 1

print("테스트 오류율 ::", testerr/len(test_x) * 100)
