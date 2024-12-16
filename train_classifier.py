import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('data_test.pickle', 'rb'))    #đọc file rb = read binary
# chuyển qua numpy vì những thư viện cần để hoạt động
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


# test_size = 0.2 => 80 train 20 test, stratify => đảm bảo phải có đủ nhãn trong x_train, y_train
# 1 nhãn có 100 bức ảnh, 3 nhãn 300 bức ảnh, x_train = 240, y_train = 240, x_test = 60, y_test = 60
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

#train model
model.fit(x_train, y_train)

#dự đoán nhãn của x_test
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('Đánh giá khả năng nhận diện: {}%'.format(score*100))


f = open('model_test.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
