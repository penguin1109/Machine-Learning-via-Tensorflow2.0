import tensorflow as tf
import numpy as np

#단일 선형 회귀(MSE)
x,y = [2,4,6,8],[81,93,91,97]
#리스트로 되어 있는 x와 y의 데이터를 numpy array로 바꾸어서 인덱스를 주어 하나씩 불러와 계산이 가능하게 한다.
x_data, y_data = np.array(x), np.array(y)
#여기서 a,b를 0,0으로 설정해주니 tape.gradinet가 작동하지 않았다.
a,b = tf.Variable(2.9),tf.Variable(0.5) #기울기와 y절편 초기화
lr = 0.01  #학습률 설정(0.05로 했는데 nan이 발생한 것으로 보아 과적합인 것 같다.)
epochs = 2001  #학습 반복 횟수 설정
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = x_data * a + b
        cost = tf.reduce_mean(tf.square(y_data - hypothesis))
    a_diff, b_diff = tape.gradient(cost, [a,b])
    a.assign_sub(lr*a_diff)
    b.assign_sub(lr*b_diff)
    if i % 100 == 0:
        print('epoch = %.f, 기울기 = %.04f, 절편 = %.04f' %(i,a,b))

#다중 선형 회귀(MSE)
data = np.array([[73.,80.,75.,152.,],[93.,88.,93.,185.,],[89.,91.,90.,180.,],[73.,66.,70.,142.,]], dtype = np.float32)
x,y = data[:,:-1], data[:,[-1]]
a,b = tf.Variable(tf.random_normal([3,1])), tf.Variable(tf.random_normal([1]))
lr, epochs = 0.00000001, 2001
for i in range(epochs):
    with tf.GradientTape() as tape:
        #행렬의 연산이기 때문에 tf.matmul을 이용한다.
        hypothesis = tf.matmul(x,a)+b
        cost = tf.reduce_mean(tf.square(hypothesis-y))
    a_diff, b_diff = tape.gradient(cost, [a,b])
    a.assign_sub(lr*a_diff)
    b.assign_sub(lr*b_diff)
    if i%100 == 0:
        print("{:5}|{:10.4f}".format(i,cost.numpy()))



#Logistic Regression(로지스틱 회귀, 분류 - Sigmoid 함수 이용)

#입력값이 2개인 로지스틱 회귀
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#다중 선형 회귀
import math
#sigmoid 함수 정의
def sigmoid(x):
    return 1/(1+math.exp(-x))
x,y,a = 1,0,tf.random.normal([1],0,1) #(shape, min, max)
b = tf.random.normal([1])
#경사 하강법을 이용한 뉴런의 학습
lr, epochs = 0.1, 2001
for i in range(epochs):
    output = sigmoid(x*a+b)
    cost = y-output
    a = a+x*lr*cost
    b = b+lr*cost
    if i%100 == 0:
        print(i,cost, output)

#입력값이 2개인 네트워크의 예(XOR 네트워크)
import tensorflow as tf
import numpy as np

x,y = np.array([[1,1],[1,0],[0,1],[0,0]]), np.array([[0], [1],[1],[0]])
model = tf.keras.Sequential()

#input_shape는 sequential 모델의 첫번째 레이어에서만 입력의 차원수를 정의하기 위해 사용됨

model.add(tf.keras.layers.Dense(units = 2, activation = 'sigmoid', input_shape = (2,)))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#최적화 함수 optimizer을 tf.keras를 이용해서 편리하게 표현 가능
#원래는 이전에 한 것 처럼 복잡한 수학식을 일일히 코딩해야 한다.
#SGD는 오차를 줄이기 위해 가중치를 업데이트하는 방법중에 경사 하강법이고
#mse는 tf.reduce_mean(tf.square(y_data-hypothesis))와 같이 평균 제곱 오차를 손실계산에 사용한 것이다

model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.1), loss ='mse')

#기존에 for문을 이용했던 것을 model.fit으로 학습 시킨다
#batch_size는 한번에 학습시키는 데이터의 개수

history = model.fit(x,y,epochs = 2001, batch_size = 1)
