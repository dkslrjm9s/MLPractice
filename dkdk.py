import requests
import os
import zipfile
import pandas as pd
import numpy as np


# mnist1_train.csv(60000건), mnist1_test.csv(10000건)
# mnist2_train.csv(60000건), mnist2_test.csv(10000건)
# result
# 784(28*28)

# result        mnist1      mnist2
# 0             0           0   T-shirt/top
# 1             1           1   Trouser
# 2             2           2   Pullover
# 3             3           3   Dress
# 4             4           4   Coat
# 5             5           5   Sandal
# 6             6           6   Shirt
# 7             7           7   Sneaker
# 8             8           8   Bag
# 9             9           9   Ankle boot


def downloadFile(ver):
    if not os.path.isfile("mnist" + str(ver) + "_test.csv"):
        remote_url = "https://github.com/AnDeoukKyi/tistory/raw/main/mlData/data_zip/mnist" + str(ver) + "_csv.zip"
        local_file = "mnist" + str(ver) + "_csv.zip"
        data = requests.get(remote_url)
        # download
        with open(local_file, 'wb') as file:
            file.write(data.content)
        # file wait
        while not os.path.isfile(local_file):
            pass
        # unzip
        zipfile.ZipFile(local_file).extractall()

        # del zipfile
        os.remove(local_file)


def mnist1Data():
    downloadFile(1)
    x_train = pd.read_csv("mnist1_train.csv", header=None).values
    x_train = x_train.astype(np.uint8)
    y_train = x_train[:, 0]
    x_train = np.delete(x_train, 0, axis=1).reshape(len(x_train), 28, 28)

    x_test = pd.read_csv("mnist1_test.csv", header=None).values
    x_test = x_test.astype(np.uint8)
    y_test = x_test[:, 0]
    x_test = np.delete(x_test, 0, axis=1).reshape(len(x_test), 28, 28)
    return x_train, y_train, x_test, y_test


def mnist2Data():
    downloadFile(2)
    x_train = pd.read_csv("mnist2_train.csv", header=None).values
    x_train = x_train.astype(np.uint8)
    y_train = x_train[:, 0]
    x_train = np.delete(x_train, 0, axis=1).reshape(len(x_train), 28, 28)

    x_test = pd.read_csv("mnist2_test.csv", header=None).values
    x_test = x_test.astype(np.uint8)
    y_test = x_test[:, 0]
    x_test = np.delete(x_test, 0, axis=1).reshape(len(x_test), 28, 28)
    return x_train, y_train, x_test, y_test


# 사용법
x_train, y_train, x_test, y_test = mnist1Data()

x_train, y_train, x_test, y_test = mnist2Data()