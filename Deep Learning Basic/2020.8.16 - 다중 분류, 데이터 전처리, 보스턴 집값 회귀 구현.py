import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#1. binary_evaluation 연습하기
#같은 결과를 반복할 수 있도록 하기 위해서 random.seed를 이용한다.
np.random.seed(3)
tf.random.set_seed(3)

#데이터를 읽어올때는 pd.read_csv()함수를 이용하고 가공할 때는 np.loadtxt이용
#csv파일은 정보가 ','를 기준으로 정렬되어 있음
df = np.loadtxt('C:\pima-indians-diabetes.csv', delimiter=',')
x_data, y_data = df[:,0:8], df[:,8]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim = 8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
#2개의 결과값으로 나뉘는 것이기 떄문에 'binary_crossentropy'를 이용한다.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_data, y_data, epochs = 200, batch_size = 10)
#예측의 정확도를 측정하는 model.evaluate를 이용
print(model.evaluate(x_data, y_data)[1])





#2. 다중 분류 연습하기 + 데이터 전처리(원핫인코딩)
np.random.seed(3)
tf.random.set_seed(3)

df = pd.read_csv('C:\iris.csv',names = ['sepal_length','sepal_width','petal_length','petal_width','species'])
dataset = df.values
x_data, y_data = dataset[:,0:4].astype(float), dataset[:,4]

#y_data가 문자열로 이루어져 있기 때문에 이 이름을 숫자 형태로 바꿔주어야 한다.
from sklearn.preprocessing import LabelEncoder
#라벨 인코더를 생성하는 과정이 불필요 할줄 알았는데 알고 보니 fit() 함수를 이용하는데 중요했다
e = LabelEncoder()
e.fit(y_data)
y_encoded = e.transform(y_data)

#활성화 함수를 적용하기 위해서는 y값이 0과1로 이루어져 있어야 함
y_encoded = tf.keras.utils.to_categorical(y_encoded)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim = 4, activation = 'relu'))
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['accuracy'])
model.fit(x_data, y_encoded, epochs = 50, batch_size = 1)

print(model.evaluate(x_data, y_encoded)[1])




#3. 과적합 피하기(k겹 교차 검증)
#k겹 교차 검증이란 학습 데이터를 k개의 데이터로 나누어서 하나씩 테스트셋으로 이용하고 나머지는 모두 합해서 학습셋으로 이용하는 것이다.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#seed, 즉 시드 값을 설정해 줌으로서 계속 같은 입력값을 반복해서 사용할 수 있다.(같은 순서로)
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('C:\sonar.csv', header = None)
dataset = df.values
x,y_data = dataset[:,0:60], dataset[:,60]

from sklearn.preprocessing import LabelEncoder
#y의 데이터를, 즉 클래스가 문자열의 형태이므로 숫자로 바꾸어 준다.
e = LabelEncoder()
e.fit(y_data)
y = e.transform(y_data)
x = np.asarray(x).astype(np.float32)
y = np.asarray(y).astype(np.float32)
#과적합 방지를 위해 'k겹 교차 검증'을 진행
#데이터셋을 k개로 나누어 하나씩 테스트셋으로 이용하고 나머지를 모두 합해서 학습셋으로 사용하는 방법
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

for train, test in skf.split(x,y):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(24, input_dim = 60, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(x[train], y[train], epochs = 100, batch_size = 5)
    print(float(model.evaluate(x[test], y[test])[1]))




#4. 데이터 성능 올리기(와인 데이터를 이용한 딥러닝 구현)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
dataset = pd.read_csv('C:\wine.csv', header = None)

#0 < frac<=1인데, 전체 데이터 중에 학습 데이터로 사용할 비율을 설정해 주는 것이다
data = dataset.sample(frac = 0.33)
data = data.values
x,y = data[:,0:12], data[:,12]
x,y = np.asarray(x).astype(np.float32), np.asarray(y).astype(np.float32)

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, input_dim = 12, activation = 'relu'))
model.add(tf.keras.layers.Dense(12, activation = 'relu'))
model.add(tf.keras.layers.Dense(8, activation = 'relu'))
model.add(tf.keras.layers.Dense(4, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

import os
modelp = './model/'
if not os.path.exists(modelp):
    os.mkdir(modelp)
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 1, save_best_only = True)
#학습 조기 중단
#일정 횟수 이상의 epoch를 수행하게 되면 학습셋의 정확도는 올라가지만 과적합에 의해 테스트셋의 실험 결과는 점점 나빠지게 된다.
#따라서 EarlyStopping()함수를 이용해서 테스트셋 오차가 줄지 않으면 학습을 멈추게 한다.
#앞서 이용한 checkpointer 과 EarlyStopping모두 tf.keras.callbacks에 해당하는 함수이다.
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 100)
#callback에 여러개의 변수를 지정하는 것은 충분히 가능하다.

#아래 validation_split의 비율은 나중에 평가용 데이터로 이용할 데이터의 비율이고 따라서 80%로 학습을 진행하게 된다.
model.fit(x,y,validation_split = 0.2, epochs = 2000, batch_size = 500,verbose = 0, callbacks = [checkpointer, early_stopping_callback])
print(model.evaluate(x,y)[1])





#5. 보스턴 집값 데이터로 선형 회귀 구현하기
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv('C:\housing.csv', delim_whitespace= True,header = None)
dataset = df.values
x,y = dataset[:, 0:13], dataset[:,13]
x = np.asarray(x).astype(np.float32)
y = np.asarray(y).astype(np.float32)
#학습 데이터와 평가 데이터를 나누어 준다.(학습 70% 평가 30%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = seed)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, input_dim = 13, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation = 'relu'))
#선형 회귀의 경우에는 마지막 layer의 활성화 함수를 정의해 줄 필요가 없다.
model.add(tf.keras.layers.Dense(1))
model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 10)

#flatten() 함수를 이용함으로서 1차원 구조로 데이터를 바꾸어준다.
y_pred = model.predict(x_test).flatten()
for i in range(10):
    print('%s, %s'%(str(y[i]), str(y_pred[i])))
#결과적으로 예상 가격과 실제 가격이 비례하면서 변화함을 알 수 있다.

