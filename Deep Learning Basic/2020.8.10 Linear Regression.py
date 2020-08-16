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
