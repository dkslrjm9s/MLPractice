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

# ??? ??? ?????? ?????? ??????
iLnum = 15 # ???????????? ??????
hLnum = 6 # ???????????? ??????
oLnum = 3 # ???????????? ??????

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

def crossEntropyError(y, t): #y: ????????? ????????? t: ??????
    delta = 1e-7 #?????? ?????? ??? (y??? 0??? ?????? -inf ?????? ??????)
    return -np.sum(t*np.log(y+delta))

def Derivative_softmax_corssentropy(y,t):
    return y-t

def Derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

for ind_epoch in range(epoch):
    print(ind_epoch + 1, "?????? ???????????????.", ind_epoch + 1, "/", epoch)
    for index_train in range(len(train_x)):
        # FeedForword(?????????)
        # ????????? ??????
        hSum = w_sum(train_x[index_train], iW, iB)
        hLayer = sigmoid(w_sum(train_x[index_train], iW, iB))

        # ????????? ??????
        oSum = w_sum(hLayer, hW, hB)
        oLayer = softmax(w_sum(hLayer, hW, hB))

        # ?????? ?????? ?????? ?????????
        error = crossEntropyError(oLayer, train_y[index_train])
        Error[ind_epoch][index_train] = error

        # Back-Propagation(?????????)
        # ???????????? ???????????? ?????? ??????
        # ??????????????? ???????????? ?????? : Derivative_softmax_corssentropy
        temphW = hW  # ????????? ????????? ?????? ??????
        a2 = Derivative_softmax_corssentropy(oLayer, train_y[index_train])
        hB = hB - learnRate * a2
        hW = hW - (learnRate * a2.reshape(a2.shape[0], 1) * hLayer)

        # ???????????? ???????????? ?????? ??????
        # ???????????? ???????????? ?????????
        # ????????? ???????????? ?????????
        a1 = np.sum(a2.reshape(a2.shape[0], 1) * temphW * Derivative_sigmoid(hSum), axis=0)
        iB = iB - learnRate * a1
        iW = iW - learnRate * a1.reshape(a1.shape[0], 1) * train_x[index_train]

    print(ind_epoch + 1, "?????? ?????? ?????? :: ", np.sum(Error[ind_epoch]) / Error.shape[1])

#????????? ??????
testerr = 0
for index_test in range(len(test_x)):
    # FeedForword(?????????)
    # ????????? ??????
    hLayer = sigmoid(w_sum(test_x[index_test], iW, iB))

    # ????????? ??????
    oLayer = softmax(w_sum(hLayer, hW, hB))
    print(oLayer)
    if (str(np.argmax(oLayer)) != str(np.argmax(test_y[index_test]))):
        testerr += 1

print("????????? ????????? ::", testerr / len(test_x) * 100)