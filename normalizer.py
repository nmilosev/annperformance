from sklearn import datasets

dataset = datasets.load_boston()
data, target = dataset.data, dataset.target

from sklearn import preprocessing

data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data = data_scaler.fit_transform(data)
target = target_scaler.fit_transform(target)


for i in range(0, len(data)):
    for j in data[i]:
        print("{0:2f},".format(j), end='')
    print("{0:2f}".format(target[i], end=''))
