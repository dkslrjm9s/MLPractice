import numpy as np

# 아이리스 데이터 로드 및 트레인 set과 test set 분리
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

# iris data 구성
# iris.feature_names : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# iris target 구성
# iris.target_names : ['setosa', 'versicolor', 'virginica']

# 학습전에 뚜렷하게 구별 가능한 데이터를 정해서 진행
setosa = None
versicolor = None
virginica = None
for ind in range(len(iris.target)):
    updateVar = iris.target_names[iris.target[ind]]

    if globals()[updateVar] is None:
        globals()[updateVar] = np.array([iris.data[ind,:]])
    else:
        globals()[updateVar] = np.append(globals()[updateVar], np.array([iris.data[ind,:]]), axis=0)

# 실제로 잘 나눠 졌는지 확인
# iris.target[iris.target==0].shape

# 각 종별로 data의 평균값 확인
# np.sum(setosa, axis=0) / setosa.shape[0]
# setosa : [5.006, 3.428, 1.462, 0.246]
# versicolor : [5.936, 2.77 , 4.26 , 1.326]
# virginica :[6.588, 2.974, 5.552, 2.026]

# 'sepal length (cm)', 'sepal width (cm)', 'petal width (cm)' 와 종('setosa', 'versicolor', 'virginica') 이 주어졌을 때 petal length (cm) 을 예측하는 회귀모델 구현
# 전처리 진행
data_x = np.concatenate([np.delete(iris.data, 2, axis=1), iris.target.reshape(iris.target.shape[0], 1)], axis = 1)
data_y = iris.data[:, 2]


def nomalization(x):
    max_var = np.max(data_y)
    min_var = np.min(data_y)
    return (x-min_var)/(max_var-min_var)

data_y = nomalization(data_y)

#train_test_split(x,y, test_size,shuffle,stratify,random_state)
#
# test_size: 테스트 셋 구성의 비율을 나타냅니다. train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다.
# shuffle: default=True 입니다. split을 해주기 이전에 섞을건지 여부입니다. 보통은 default 값으로 놔둡니다.
# stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값입니다. stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. (한 쪽에 쏠려서 분배되는 것을 방지합니다) 만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.
# random_state: 세트를 섞을 때 해당 int 값을 보고 섞으며, 하이퍼 파라미터를 튜닝시 이 값을 고정해두고 튜닝해야 매번 데이터셋이 변경되는 것을 방지할 수 있습니다.
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=42)

# 층 별 노드 갯수 셋팅
iLnum = 4 # 입력층의 갯수
hLnum = 2 # 은닉층의 갯수
oLnum = 1 # 출력층의 갯수

# 층 별 기본 설정 셋팅
#입력층
# input - hidden layer
# w : 다음층 갯수 by 현재 층 갯수 행렬 (h by w / y by x)
# b : 다음층 갯수
# h(actuvation) : 시그모이드 함수
iW = np.random.randn(hLnum,iLnum)
# array([[13375257.59078464,  6556500.32489677,  7867797.9684276 ,
#          2884858.98187539],
#        [  -68181.1008168 ,   -33422.00326687,   -40106.03075334,
#           -14705.72825433]])
iB = np.random.randn(hLnum)
# array([2622600.9168694 ,  -13368.31839451])
#은닉층
# hidden - output layer
# w : 다음층 갯수 by 현재 층 갯수 행렬 (h by w / y by x)
# b : 다음층 갯수
# h(actuvation) : 렐루 함수
hW = np.random.randn(oLnum,hLnum)
hB = np.random.randn(oLnum)

#출력층 3개
# 손실함수 : MSE

epoch = 200
learnRate = 0.7
Error = np.zeros((epoch, train_x.shape[0]), dtype="float")

#axis가 1 이면 행끼리 더하는 것이고, axis가 0 이면 열끼리 더하는 것이다.
def w_sum(x, w, b):
    return np.sum(x * w, axis=1) + b

def sigmoid(x):
    # overflow 방지(너무작아서)
    # x = np.clip(x, -500, 500)
    # return 1 / (1 - np.exp(-x))

    # if -x > np.log(np.finfo(type(x[0])).max):
    #     return 0.0
    # a = np.exp(-x)
    # return 1.0 / (1.0 + a)
    return 1 / (1 + np.exp(-x))


def Derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def Derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def leaky_relu(x):
    return np.maximum(0.01 * x,x)

def Derivative_leaky_relu(x):
    x[x <= 0] = 0.01
    x[x > 0] = 1
    return x

def MSE(y, t): #y: 신경망 출력값 t: 정답
    return (1/2)*np.sum((t-y)**2)

def Derivative_MSE(y,t):
    return -(t-y)




for ind_epoch in range(epoch):
    for index_train in range(len(train_x)):
        # FeedForword(순전파)
        # 은닉층 구현
        hSum = w_sum(train_x[index_train], iW, iB)
        hLayer = sigmoid(hSum)

        # 출력층 구현
        oSum = w_sum(hLayer, hW, hB)
        oLayer = sigmoid(oSum)

        # 실제 값과 오차 구하기
        error = MSE(oLayer, train_y[index_train])

        # Back-Propagation(역전파)
        # 은닉층의 가중치와 편향 갱신
        # 손실함수와 비용함수 미분 : Derivative_MSE
        temphW = hW  # 다음층 갱신을 위해 저장
        a2 = Derivative_MSE(oLayer, train_y[index_train]) * Derivative_sigmoid(oSum)
        hB = hB - learnRate * a2
        hW = hW - (learnRate * a2.reshape(a2.shape[0], 1) * hLayer)

        # 입력층의 가중치와 편향 갱신
        # 은닉층을 갱신했던 미분값
        # 갱신전 은닉층의 가중치
        a1 = np.sum(a2.reshape(a2.shape[0], 1) * temphW * Derivative_sigmoid(hSum), axis=0)
        iB = iB - learnRate * a1
        iW = iW - (learnRate * a1.reshape(a1.shape[0], 1) * train_x[index_train])

        Error[ind_epoch][index_train] = error

    print(ind_epoch + 1, "번째 학습입니다.", ind_epoch + 1, "/", epoch)
    print(ind_epoch + 1, "번째 학습 MSE :: ", np.sum(Error[ind_epoch]) / Error.shape[1])

print("=" * 50)
# 테스트 시작
testerr = []

for index_test in range(len(test_x)):
    # FeedForword(순전파)
    # 은닉층 구현
    hLayer = sigmoid(w_sum(test_x[index_test], iW, iB))

    # 출력층 구현
    oLayer = sigmoid(w_sum(hLayer, hW, hB))

    print("oLayer[0]", oLayer[0])
    print("test_y[index_test]", test_y[index_test])
    print("=" * 50)
    testerr.append(MSE(oLayer[0], test_y[index_test]))


# 렐루로 하면,,, 마지막 출력층 값이 다 똑같이 나오는 현상이 일어난다.
# 가중치에 0의 값이 계속 더해 지며 너무 값이 작게 갱신되는 건지 대부분 0.5에 멈추는 현상이 나온다...
# 이건 방법을 한번 찾아봐야 겠다...

print("테스트 MSE 평균 ::", sum(testerr) / len(testerr))